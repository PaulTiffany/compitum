Release Checklist: v0.1.1

Goal
- Ship compitum v0.1.1 with clear, reproducible evidence aligned to Lyapunov/geometry claims and RouterBench results.

Quality gates (must be green)
- Core CI (matrix: ubuntu-latest, windows-latest)
  - Ruff lint
  - MyPy types (package)
  - Import smoke
  - Unit + property tests (100% coverage on src/compitum)
- Rigor workflow
  - MyPy strict (src/compitum)
  - Bandit (src + examples + scripts)
  - Invariants suite
  - Benchmarks JSON artifact
  - Docs build (HTML, nitpicky) and linkcheck
- RouterBench
  - Dataset present via cache/release asset/fetch script
  - Local evaluation passes with capped evals in PRs; full sweep in nightly/manual
  - Compitum evaluation + HTML analysis report uploaded as artifact

Artifacts to publish (attach to GitHub Release)
- Wheels and sdist from release workflow
- Reports: reports/report_release.html; RouterBench analysis (HTML/CSV/PKL)
- LLM snapshot: docs/repo_snapshot.jsonl
- Pretrained bundle (if applicable): metrics/constraints snapshot with schema + checksums

Provenance and reproducibility
- Record training config, seed, environment, and dataset checksum for any pretrained bundle
- Include CHECKSUMS.txt for all attached artifacts
- Ensure tools/verify_repro.py passes on CI (informational) and in local reproduction

Docs updates
- README: Rigor Levels + RouterBench label instructions
- WORKFLOWS.md: CI/Extended/Full breakdown, RouterBench cache/fetch notes
- Artifact-README.md: where to find and how to use attached artifacts

Tagging and release
- Bump version in pyproject.toml if needed
- Tag: v0.1.1
- Run Release workflow on tag
- Verify smoke install from wheel

Post-release
- Update compitum.space (Pages) to reflect latest docs and reports
- Announce with links to Release, Docs, and Reports

