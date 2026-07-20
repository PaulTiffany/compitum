import pandas as pd

from tools.generate_eval_tables import (
    compute_per_baseline_winrate,
    compute_frontier_gap,
    compute_panel_summary,
)


def _toy_df() -> pd.DataFrame:
    rows = []
    # Two tasks, compitum + baseline, two WTPs
    for eval_name in ["t1", "t2"]:
        for w in [0.1, 1.0]:
            rows.append(
                {
                    "eval_name": eval_name,
                    "model_name": "compitum",
                    "willingness_to_pay": w,
                    "performance": 0.6 if eval_name == "t1" else 0.55,
                    "total_cost": 0.2,
                }
            )
            rows.append(
                {
                    "eval_name": eval_name,
                    "model_name": "baselineX",
                    "willingness_to_pay": w,
                    "performance": 0.58 if eval_name == "t1" else 0.5,
                    "total_cost": 0.25,
                }
            )
    return pd.DataFrame(rows)


def test_winrate_and_panel_tables_shape():
    df = _toy_df()
    out = compute_per_baseline_winrate(df, [0.1, 1.0])
    assert "| Baseline | WTP | Win Rate | N |" in out or out.startswith("No comparable")
    panel = compute_panel_summary(df, [0.1, 1.0])
    assert "# Panel Summary" in panel


def test_frontier_gap_monotone_header():
    df = _toy_df()
    txt = compute_frontier_gap(df, [0.1, 1.0], bootstrap=0)
    assert "| WTP | Avg Gap to Frontier" in txt
