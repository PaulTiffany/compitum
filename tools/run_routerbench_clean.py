import os
import sys
import runpy
from pathlib import Path


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
        # As a last resort, fall back to character count heuristic to avoid crashes.
        return len(s if isinstance(s, str) else str(s))


def main() -> None:
    # Parse optional tokenizer backend flag (default tiktoken)
    backend = os.environ.get("RB_TOKENIZER_BACKEND", "tiktoken").lower()
    for i, arg in list(enumerate(sys.argv)):
        if arg.startswith("--tokenizer-backend="):
            _, v = arg.split("=", 1)
            backend = v.lower()
            sys.argv.pop(i)
            break

    # Suppress noisy stdout warnings from tokencost
    try:
        import tokencost.costs as _tc_costs

        _tc_costs.print = _no_print
    except Exception:
        pass

    # Standardize token counting to tiktoken for reproducibility
    try:
        import tokencost as _tc

        if backend == "tiktoken":
            _tc.count_string_tokens = _tiktoken_count_string_tokens  # type: ignore[attr-defined]
        elif backend == "tokencost":
            pass  # use library defaults
        elif backend == "hf":
            def _hf_count_string_tokens(s: str, model_name: str) -> int:
                try:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                    return len(tok.encode(s if isinstance(s, str) else str(s)))
                except Exception:
                    return _tiktoken_count_string_tokens(s, model_name)

            _tc.count_string_tokens = _hf_count_string_tokens  # type: ignore[attr-defined]
        else:
            _tc.count_string_tokens = _tiktoken_count_string_tokens  # type: ignore[attr-defined]
    except Exception:
        pass

    # Ensure RouterBench's script-style absolute imports resolve
    project_root = next(
        (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
        Path(__file__).resolve().parents[1],
    )
    rb_pkg_dir = project_root / "src" / "routerbench"
    if str(rb_pkg_dir) not in sys.path:
        sys.path.insert(0, str(rb_pkg_dir))

    # Provide default connection string expected by upstream evaluate script
    os.environ.setdefault("CONNECTION_STRING", "mongodb://localhost:27017/")

    # Patch Windows-unsafe filename generation in evaluate_utils.save_results
    try:
        import evaluate_utils as _eval_utils  # type: ignore
        from datetime import datetime

        def _safe_save_results(result_df, train_test_split: bool = False, eval_name: str = ""):
            date_string = datetime.now().strftime("%b%d-%H-%M")
            output_filename = (
                f"data/eval_results/eval_results-eval-{eval_name}-{date_string}"
                f"{'-val_split' if train_test_split else ''}.csv"
            )
            result_df.to_csv(output_filename, index=False)
            print("Saved to: ", output_filename)
            return output_filename

        _eval_utils.save_results = _safe_save_results  # type: ignore[attr-defined]
    except Exception:
        pass

    # Optional: limit evaluation set size for faster iteration
    try:
        max_evals = int(os.environ.get("RB_MAX_EVALS", "0"))
    except ValueError:
        max_evals = 0
    if max_evals > 0:
        try:
            import pandas as _pd

            _orig_read_csv = _pd.read_csv
            _orig_read_pickle = _pd.read_pickle

            def _wrap_limit_df(df):
                # Keep deterministic head for reproducibility
                return df.head(max_evals)

            def _read_csv_limited(*a, **k):
                df = _orig_read_csv(*a, **k)
                return _wrap_limit_df(df)

            def _read_pickle_limited(*a, **k):
                df = _orig_read_pickle(*a, **k)
                return _wrap_limit_df(df)

            _pd.read_csv = _read_csv_limited  # type: ignore
            _pd.read_pickle = _read_pickle_limited  # type: ignore
        except Exception:
            pass

    # Forward to RouterBench CLI entry with passed-through args
    argv = sys.argv[:]
    sys.argv = ["routerbench.evaluate_routers"] + argv[1:]
    runpy.run_module("routerbench.evaluate_routers", run_name="__main__")


if __name__ == "__main__":
    main()
