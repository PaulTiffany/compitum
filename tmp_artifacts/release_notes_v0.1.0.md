# Compitum v0.1.0 — Release Notes

Highlights

- Geometrically-aware routing with SPD metrics, constraint-first selection, and Lyapunov-inspired trust-region updates.
- Mechanistic routing certificate with utility components, feasibility, boundary diagnostics, and drift status.
- Clean documentation set (MyST + Furo), audience bridges (cs.LG/CL/SY/stat.ML), and an evidence-focused peer-review pack.

What’s New Since Pre-Release

- Documentation polish: removed encoding glitches, added a concise Math-Brief summary box, and rewrote CLI/Control pages for clarity.
- Certificate schema alignment: `constraints.shadow_prices` is an object map; JSON uses `boundary` and `drift` field names.
- Future directions page outlining optional “dual shadow pricing” via a lightweight online primal–dual controller (default-off).

How to Verify (Windows)

```bat
make lint && make mypy && make test && make docs && make bandit
make peer-review
python -m sphinx -b html docs docs\_build\html
```

Artifacts

- Reports: `reports/report_release.html`, `reports/fixed_wtp_summary.{json,md}`, `reports/routerbench_report.md`, `reports/artifact_manifest.json`
- Docs site: `docs/_build/html` (sitemap enabled)

Compatibility

- Python 3.10–3.13 on Windows/macOS/Linux
- No judge-model dependency; offline mode and audit records supported

Future Directions (Post-Launch)

- Dual shadow pricing (true duals λ) via online primal–dual controller (observe-only or priced routing behind a flag)
- Robust boundary sensitivity (scaled epsilon, critical relaxation)
- Optional PyLantern integration for observer-bounded windows and typed surfaces

Acknowledgements

Thanks to reviewers and contributors across cs.LG, cs.CL, cs.SY, and stat.ML for helping shape a rigorous, reproducible, and accessible artifact.

