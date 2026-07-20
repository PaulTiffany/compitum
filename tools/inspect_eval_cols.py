import glob
import pandas as pd

p = sorted(glob.glob("data/rb_clean/eval_results/eval_results-eval-all-*-val_split.csv"))[-1]
df = pd.read_csv(p)
for c in list(df.columns)[:120]:
    print(c)
