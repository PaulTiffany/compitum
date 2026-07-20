import glob
import pandas as pd

p = sorted(glob.glob("data/rb_clean/eval_results/eval_results-eval-all-*-val_split.csv"))[-1]
df = pd.read_csv(p)
print("rows", len(df))
print("unique model_name count", df["model_name"].nunique())
print("first 40 names:")
for n in sorted(df["model_name"].astype(str).unique())[:40]:
    print(n)
print(
    "contains compitum?",
    any(s.startswith("compitum|") for s in df["model_name"].astype(str).unique()),
)
