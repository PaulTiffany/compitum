---
title: Results Summary
---

# Results Summary

Canonical table with fixed budgets (WTP), bootstrap 95% confidence intervals, and seeds.

| Task              | Budget (WTP) | Best Baseline Utility | Compitum Utility | Delta | 95% CI (Delta) | Seeds |
|---                |---:          |---:                   |---:              |---:   |---             |---:   |
| grade-school-math | 0.1          | -                     | -                | -     | [-, -]         | n=    |
| hellaSWAG         | 0.1          | -                     | -                | -     | [-, -]         | n=    |
| MBPP              | 0.1          | -                     | -                | -     | [-, -]         | n=    |
| Panel Avg         | 0.1          | -                     | -                | -     | [-, -]         | n=    |
| grade-school-math | 1.0          | -                     | -                | -     | [-, -]         | n=    |
| hellaSWAG         | 1.0          | -                     | -                | -     | [-, -]         | n=    |
| MBPP              | 1.0          | -                     | -                | -     | [-, -]         | n=    |
| Panel Avg         | 1.0          | -                     | -                | -     | [-, -]         | n=    |

Notes
- Baselines: KNN/MLP/cascade and RouterBench common routers. "Best Baseline" is per-evaluation unit at fixed WTP.
- CIs: nonparametric bootstrap (1,000 resamples) over evaluation units. Seeds fixed; Hypothesis derandomized.
- Reproduce via scripts/run_peer_review.bat (Windows) or Make target `peer-review` (below).

## Reproduce

Windows (one-shot):

```bat
scripts\run_peer_review.bat
```

Make (Windows):

```bash
make peer-review
```

