# LLM Usage

Compitum plays nicely with LLMs by emitting structured JSON certificates and by keeping core logic auditable. To give an LLM rich context about the repo without noise, use the lean snapshot.

## Quick: Use the Snapshot

- Download or point models to: `https://compitum.space/docs/repo_snapshot.jsonl`
- Recommended note: "Use JSONL lines with `type`, `path`, `content`. Build an index by `path` and cite `path:line` in answers."

## Snapshot with Paleae

```bash
# In the repo root
curl -fsSL https://raw.githubusercontent.com/PaulTiffany/paleae/main/paleae.py -o paleae.py
python paleae.py --profile ai_optimized \
  --include "^src/compitum/" \
  --include "^(README|PHILOSOPHY|SECURITY|CONTRIBUTING|SUPPORT|CHANGELOG)\.md$" \
  --include "^docs/(Getting-Started|CLI|Invariants|Philosophy|API-Reference|LLM-Usage|ACCESSIBILITY)\.md$"
```

Tip: Add exclusions to `.paleaeignore` to skip caches/artifacts: `.venv*`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.hypothesis/`, `data/`, `artifacts/`, `reports/`.

## Useful Entry Points for LLMs

- `src/compitum/router.py:1` — routing loop and certificate
- `src/compitum/energy.py:1` — utility components, invariants
- `src/compitum/constraints.py:1` — feasibility and shadow prices
- `src/compitum/boundary.py:1` — tie/boundary diagnostics
- `src/compitum/control.py:1` — trust region and drift
- `PHILOSOPHY.md:1` — instantaneous RL, geometry, constraints
- `docs/ACCESSIBILITY.md:1` — a11y standards for docs and outputs

## Traces for Verification

```bash
compitum route --prompt "..." --trace > trace.json
```

Traces include `utility_components`, `constraints`, `boundary_analysis`, and `drift_status` so an LLM can reason with mechanistic signals instead of prose.

## Build a Size-Budgeted Context Pack

Use the helper to create a compact snapshot that includes `.paleaeignore` and core code/docs:

```bash
python tools/build_llm_context_pack.py --out compitum_context.jsonl --target-size 1MB
```

Adjust `--target-size` (e.g., `800KB`, `2MB`). The script ensures `.paleaeignore` is present and trims low-priority files if needed.
