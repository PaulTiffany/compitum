# Handoff capsule: Bellman-consistent discrete shadow-charge pricing

## What this is

This folder is a **packaging and provenance capsule**, not a duplicate
implementation. It exists to let a future decision — standalone
repository, module inside Sketched, structured source for
`notebook_compiler`, or a book section plus executable artifact — be
made later, without disturbing the completed scientific record, and
without anyone needing to re-read seven tranche reports to find the
canonical code.

Nothing in this folder is executable research code. It contains:

- `CLAIMS.md` — what is proved, what is empirically supported, what is
  a negative finding, what is a limitation, and what is explicitly not
  claimed.
- `MODULE_MAP.md` — a concept-to-canonical-source-path index, so a
  future extraction (e.g. `git subtree split`) knows exactly which files
  to pull without guessing.
- `NARRATIVE_SOURCE.md` — clean, presentation-agnostic prose describing
  the work, structured for ingestion by `notebook_compiler` or
  adaptation into Sketched/book source.
- `artifact_manifest.json` — a machine-readable index of the canonical
  repository, branch, commit, tag, source modules, reports, ADRs, and
  SHA-256 hashes of the decisive artifacts.
- `MERGE_INVENTORY.md` — the classification of what is a genuine
  production merge candidate, what is research-only, and what was
  explicitly rejected for activation.

## What remains canonical

**Nothing here is a second copy of the implementation.** The canonical
code lives, unchanged, in its existing package paths on branch
`experiment/fabricpc-trajectory-observer` at the frozen commit tagged
`fabricpc-compitum-shadow-pricing-v1`:

```text
src/compitum/regret_lab/           -- Bellman oracles, shadow-charge pricing, belief estimators
experiments/fabricpc/              -- pilot scripts, reports, JSON artifacts, FabricPC adapters
docs/adr/0008-*.md, 0009-*.md      -- architecture decision records
```

If this capsule and the canonical repository ever disagree, the
canonical repository at the tagged commit is authoritative. This
capsule should be regenerated (hashes, paths) if the canonical record
is ever amended — it is a snapshot, not a live mirror.

## How to use this later

- To extract a standalone package: use `MODULE_MAP.md` to identify
  exactly which files to `git subtree split` or copy, and
  `artifact_manifest.json` for their hashes to verify the extraction
  matches this record.
- To feed `notebook_compiler`: `NARRATIVE_SOURCE.md` is already
  structured into stable, presentation-agnostic sections.
- To mirror into Sketched: `MODULE_MAP.md` plus `artifact_manifest.json`
  give the boundary; do not copy source into Sketched directly from this
  capsule, copy it from the canonical paths listed in `MODULE_MAP.md`.
- To decide what merges into `main`: read `MERGE_INVENTORY.md` first.
  As of this capsule, only one fix is a genuine production candidate.

## Provenance

- Canonical repository: `git@github.com:PaulTiffany/compitum.git`
- Canonical branch: `experiment/fabricpc-trajectory-observer`
- Frozen tag: `fabricpc-compitum-shadow-pricing-v1`
- Frozen commit: `617f8979daa921d326301266e55740c0746ab95c`
- This capsule was authored on submission branch: `submission/bellman-shadow-pricing`
