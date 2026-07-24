# Submission validation record

Recorded at the point of submission, on branch
`submission/bellman-shadow-pricing`, commit
`439c0d23885127e2be72723a54bfafb4c0ed476c` (built on top of the frozen
research commit `617f8979daa921d326301266e55740c0746ab95c`, tag
`fabricpc-compitum-shadow-pricing-v1`).

## 1. Full test suite

```text
.venv/Scripts/python.exe -m pytest -q -m "not routerbench and not heavy_bench"
905 passed, 1 skipped, 3 deselected
```

The 1 skip is a documented, pre-existing Windows subprocess/asyncio
issue (`tests/tools/test_examples_run_helper.py:29`), unrelated to this
work. The 3 deselections are the `routerbench` marker, out of scope.

## 2. mypy --strict

```text
.venv/Scripts/python.exe -m mypy -p compitum --ignore-missing-imports --hide-error-context
Success: no issues found in 72 source files
```

## 3. ruff

```text
.venv/Scripts/python.exe -m ruff check .
All checks passed!
```

## 4. Minimal submission demo

```text
.venv/Scripts/python.exe experiments/fabricpc/submission_demo.py
```

Output:

```text
scalar choice (first 5 steps):        ['conserve', 'spend', 'conserve', 'conserve', 'spend']
shadow-charge choice (first 5 steps):  ['defer', 'defer', 'defer', 'defer', 'spend']
online-optimal choice (first 5 steps): ['defer', 'defer', 'defer', 'defer', 'spend']
shadow-charge equality: PASS
belief boundary: remaining_steps=1 budget=4.5 observed=True
low-belief (0.05) action:  spend
high-belief (0.2) action: opportunity

overall: PASS
```

Exit code `0`. Runtime ~0.3 seconds. No FabricPC/JAX dependency, no
retraining.

## 5. JSON parsing for every submitted machine-readable artifact

All four parsed successfully with the standard library `json` module:

- `handoff/bellman_shadow_pricing/artifact_manifest.json` — valid
- `experiments/fabricpc/tranche6_5/artifacts/shadow_charge_pilot_report.json` — valid
- `experiments/fabricpc/tranche7/artifacts/gate0_report.json` — valid
- `experiments/fabricpc/tranche7/artifacts/ten_arm_pilot_report.json` — valid

## 6. Hash verification for the manifest

Every SHA-256 hash recorded in
`handoff/bellman_shadow_pricing/artifact_manifest.json` was recomputed
directly from the referenced file and compared:

```text
reports.final_synthesis:      MATCH
reports.tranche_6_5_report:   MATCH
reports.tranche_7_report:     MATCH
mrr.tranche_6_5:               MATCH
mrr.gate0:                     MATCH
mrr.ten_arm:                   MATCH
adr_0008:                      MATCH
adr_0009:                      MATCH
dependency_receipt:             MATCH

ALL HASHES MATCH
```

## 7. Link/path validation from SUBMISSION.md

14 repository-relative paths referenced in `SUBMISSION.md` (canonical
commands, artifact-map table, reproduction section) were extracted and
checked for existence on disk. All 14 resolved. No broken references.

## 8. git status

```text
On branch submission/bellman-shadow-pricing
nothing to commit, working tree clean
```

Confirmed before and after the submission-surface commit; no
uncommitted changes remain.

## 9. Remote branch and tag verification (after push)

```text
git ls-remote origin
617f8979daa921d326301266e55740c0746ab95c   refs/heads/experiment/fabricpc-trajectory-observer
80f55f36cf0f4635a7191e03b44a0810ce112c7c   refs/heads/main
439c0d23885127e2be72723a54bfafb4c0ed476c   refs/heads/submission/bellman-shadow-pricing
5a1b3ad3b7c7984b0ca69db7a4a10a90c33a6aa8   refs/tags/fabricpc-compitum-shadow-pricing-v1
617f8979daa921d326301266e55740c0746ab95c   refs/tags/fabricpc-compitum-shadow-pricing-v1^{}
```

Confirmed:

- The research branch (`experiment/fabricpc-trajectory-observer`) is on
  origin at exactly `617f8979daa921d326301266e55740c0746ab95c`.
- The annotated tag `fabricpc-compitum-shadow-pricing-v1` dereferences
  (`^{}`) to that exact same commit.
- The submission branch (`submission/bellman-shadow-pricing`) is on
  origin at `439c0d23885127e2be72723a54bfafb4c0ed476c`, built directly
  on the tagged commit with documentation/packaging additions only.
- **`main` is unchanged** at `80f55f36cf0f4635a7191e03b44a0810ce112c7c`
  — nothing from this program has been merged into it.

## Summary

All nine validation checks passed. The research record is frozen,
tagged, and pushed; the submission surface is committed, validated, and
pushed on its own branch; `main` was never touched.
