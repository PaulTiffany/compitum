import glob
import json
from pathlib import Path
import pandas as pd

p = sorted(glob.glob("data/rb_clean/eval_results/eval_results-eval-all-*-val_split.csv"))[-1]
df = pd.read_csv(p)
# Expect long format with columns: eval_name, model_name, performance, total_cost, embedding, fraction, willingness_to_pay
# Separate compitum and baselines
is_comp = df["model_name"].astype(str).str.startswith("compitum|")
is_baseline = df["model_name"].astype(str).str.match(r"^(svm\||mlp\||knn\||cascading router\|)")
comp = df[is_comp].copy()
base = df[is_baseline].copy()

# For each baseline type and WTP, compute agreement where compitum's performance >= baseline performance at same eval_name and WTP
rows = []
for base_type in ["svm", "mlp", "knn", "cascading router"]:
    bsub = base[base["model_name"].str.startswith(base_type + "|")]
    if bsub.empty:
        continue
    for w in sorted(bsub["willingness_to_pay"].dropna().unique()):
        B = bsub[bsub["willingness_to_pay"] == w]
        C = comp[comp["willingness_to_pay"] == w]
        if C.empty or B.empty:
            continue
        # Merge on eval_name
        M = pd.merge(
            C[["eval_name", "performance"]],
            B[["eval_name", "performance"]],
            on="eval_name",
            suffixes=("_comp", "_base"),
        )
        if len(M) == 0:
            continue
        win_rate = float((M["performance_comp"] >= M["performance_base"]).mean())
        rows.append(
            {
                "baseline": base_type,
                "wtp": float(w),
                "compitum_vs_baseline_win_rate": win_rate,
                "n": int(len(M)),
            }
        )

out = {"per_baseline_win_rate": rows, "csv": p}
Path("reports/per_baseline_win_rate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
