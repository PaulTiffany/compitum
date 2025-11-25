Data directory

This folder holds local data and caches used by development and benchmarking. Large or generated files are intentionally ignored by Git to keep the repository lean and license‑safe.

RouterBench 5‑shot file
- Not versioned here. Fetch from: https://huggingface.co/datasets/withmartian/routerbench/blob/main/routerbench_5shot.pkl
- Recommended: use the helper script to download and verify, and optionally copy into the legacy location:

```
python scripts/fetch_routerbench.py --also-copy-to-src
```

Common subfolders (ignored)
- `eval_results/`: outputs from local runs
- `pretrain_predictors/`: pre‑trained predictor snapshots
- `rb_clean/`, `rb_fast/`, `routerbench/`: upstream RouterBench data/products
- `embedding_cache_*.pkl`: local embedding caches

Security note
Pickle files (`.pkl`) can execute code when loaded. Only fetch from trusted sources and verify checksums when possible.




Data policy
- See docs/Data-Policy.md for CI/CD boundaries, security, and provenance guidance.

