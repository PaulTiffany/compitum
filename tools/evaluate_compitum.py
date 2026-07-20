import os
import sys
from pathlib import Path
from datetime import datetime
import runpy

import pandas as pd
import yaml


def _no_print(*args, **kwargs):
    pass


def _tiktoken_count_string_tokens(s: str, model_name: str) -> int:
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s if isinstance(s, str) else str(s)))
    except Exception:
        return len(s if isinstance(s, str) else str(s))


def main() -> None:
    project_root = next(
        (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
        Path(__file__).resolve().parents[1],
    )
    src_dir = project_root / "src"
    rb_dir = src_dir / "routerbench"

    # Ensure imports for both routerbench and compitum
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(rb_dir) not in sys.path:
        sys.path.insert(0, str(rb_dir))

    # Quiet tokencost, and standardize tokenizer for consistency
    try:
        import tokencost.costs as _tc_costs

        _tc_costs.print = _no_print
    except Exception:
        pass
    try:
        import tokencost as _tc

        _tc.count_string_tokens = _tiktoken_count_string_tokens  # type: ignore[attr-defined]
    except Exception:
        pass

    # Default env for Mongo conn (unused when local cache is enabled)
    os.environ.setdefault("CONNECTION_STRING", "mongodb://localhost:27017/")

    # Load config
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(project_root / "data" / "rb_clean" / "evaluate_routers.yaml"),
    )
    ap.add_argument(
        "--max-evals", type=int, default=0, help="Optional cap on number of eval rows (head)"
    )
    ap.add_argument(
        "--wtp-list",
        type=str,
        default="0.0001,0.001,0.01,0.1,1.0,10.0",
        help="Comma-separated willingness_to_pay values (e.g. '0.0001,0.001,0.01,0.1,1.0,10.0')",
    )
    ap.add_argument(
        "--filter-eval", type=str, default=None, help="Optional single eval_name to filter"
    )
    args, unknown = ap.parse_known_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_path = cfg.get(
        "data_path", str(project_root / "src" / "routerbench" / "routerbench_5shot.pkl")
    )
    data_name = cfg.get("data_name", "rb_clean_compitum")
    wanted_eval_name = cfg.get("wanted_eval_name")
    train_fraction = float(cfg.get("train_fraction", 0.7))
    local_cache = bool(cfg.get("local_cache", True))
    embedding_models = cfg.get("embedding_models", ["all-MiniLM-L12-v2"]) or ["all-MiniLM-L12-v2"]
    embedding_model = embedding_models[0]

    # Load dataset
    if str(data_path).endswith(".csv"):
        dataset_df = pd.read_csv(data_path)
    else:
        dataset_df = pd.read_pickle(data_path)
    if args.filter_eval:
        dataset_df = dataset_df[dataset_df["eval_name"].astype(str) == args.filter_eval]
        print(f"Filtered dataset to eval_name == {args.filter_eval} (rows={len(dataset_df)})")
    if args.max_evals and args.max_evals > 0:
        dataset_df = dataset_df.head(args.max_evals)

    # Bring in RouterBench utilities
    from routerbench.utils import get_models_to_route, WILLINGNESS_TO_PAY
    from routerbench.evaluate_utils import (
        combined_eval_results_to_eval_collection,
        save_results as rb_save_results,
    )
    import routerbench.evaluate_routers as rb_eval

    MODELS_TO_ROUTE = get_models_to_route(dataset_df)

    # Build our Compitum adapter
    from tools.routerbench.routers.compitum_router import CompitumRouterAdapter

    # Use pretrained predictors if available to avoid lengthy fitting
    pretrained_path = (
        project_root / "data" / "pretrain_predictors" / "predictors_all-MiniLM-L12-v2_0.1.joblib"
    )
    adapter = CompitumRouterAdapter(
        router_defaults_path="configs/router_defaults.yaml",
        constraints_path="configs/constraints_routerbench_default.yaml",
        data_path="src/routerbench/routerbench_5shot.pkl",
        pretrained_predictors_path=pretrained_path if pretrained_path.exists() else None,
    )

    # Build param name following RouterBench convention for later parsing
    compitum_name = f"compitum|embedding:{embedding_model}|fraction:{train_fraction}"
    model_routers_and_names = [(adapter, compitum_name)]

    # Run evaluation using RouterBench machinery (local mode)
    # Wire in MODELS_TO_ROUTE expected by upstream helper
    rb_eval.MODELS_TO_ROUTE = MODELS_TO_ROUTE  # type: ignore[attr-defined]
    # Override WTP grid if provided
    try:
        wtp_list = [float(x.strip()) for x in args.wtp_list.split(",") if x.strip()]
    except Exception:
        wtp_list = [1.0]
    rb_eval.WILLINGNESS_TO_PAY = wtp_list  # type: ignore[attr-defined]
    print(f"Using WTP grid: {wtp_list}")

    result_df = rb_eval.get_results_for_all_evals(
        dataset_df,
        model_routers_and_names=model_routers_and_names,
        use_local=True,
    )

    # Safe save (Windows-friendly) for per-eval CSV, then collection
    ts = datetime.utcnow().strftime("%m-%d-%H-%M").replace(":", "-")
    per_eval_path = project_root / "data" / data_name / "eval_results"
    per_eval_path.mkdir(parents=True, exist_ok=True)
    label = args.filter_eval or (wanted_eval_name or "all")
    per_eval_csv = per_eval_path / f"eval_results-eval-{label}-{ts}-val_split.csv"
    result_df.to_csv(per_eval_csv, index=False)
    print(f"Saved to: {per_eval_csv}")

    # Build combined collection (inject compitum schema at runtime)
    try:
        import routerbench.evaluation.utils as eval_utils
        import routerbench.evaluation.eval as eval_mod

        # Restrict transformation schema to compitum only to avoid KeyErrors
        # when baseline routers are not present in the results.
        compitum_schema = {"compitum": ["embedding", "fraction", "willingness_to_pay"]}
        eval_utils.model_metric_dict.clear()
        eval_utils.model_metric_dict.update(compitum_schema)
        # Keep eval module's reference in sync (it captured the dict at import time)
        eval_mod.model_metric_dict.clear()
        eval_mod.model_metric_dict.update(compitum_schema)

        combined_eval_results_to_eval_collection(
            [str(per_eval_csv)],
            data_name=data_name,
            models_to_route=MODELS_TO_ROUTE,
        )
    except Exception as e:
        print("Warning: could not build EvaluationResultCollection for compitum.")
        print(f"Per-eval CSV saved at: {per_eval_csv}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
