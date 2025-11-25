import json
from pathlib import Path
import pandas as pd

def main() -> int:
    base = Path('data/rb_clean/eval_results')
    all_csvs = sorted(base.glob('eval_results-eval-all-*-val_split.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not all_csvs:
        print('No eval_results-eval-all CSV found')
        return 1
    path = all_csvs[0]
    df = pd.read_csv(path)

    def choice_cols(prefix: str):
        return sorted([c for c in df.columns if c.startswith(prefix) and ('|willingness_to_pay:' in c) and not c.endswith('|total_cost')])

    res: dict[str, list[tuple[float, float]]] = {}
    for base_prefix in ['svm|', 'mlp|', 'knn|', 'cascading router|']:
        bcols = choice_cols(base_prefix)
        if not bcols:
            continue
        ccols = choice_cols('compitum|')
        if not ccols:
            continue
        def wtp_of(col: str):
            try:
                return float(col.split('|willingness_to_pay:')[-1])
            except Exception:
                return None
        bmap = {wtp_of(c): c for c in bcols}
        cmap = {wtp_of(c): c for c in ccols}
        shared_wtps = sorted(set([k for k in bmap.keys() if k is not None]) & set([k for k in cmap.keys() if k is not None]))
        if not shared_wtps:
            continue
        wins: list[tuple[float, float]] = []
        for w in shared_wtps:
            bc = bmap[w]
            cc = cmap[w]
            agree = float((df[bc] == df[cc]).mean())
            wins.append((float(w), agree))
        res[base_prefix.rstrip('|')] = wins

    out = {'per_baseline_choice_agreement': res, 'csv': str(path)}
    Path('reports/per_baseline_agreement.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
