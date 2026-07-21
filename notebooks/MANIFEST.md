---
title: Notebook Manifest
description: Authoritative classification of every notebook in this repository, and what validation each class receives.
---

# Notebook Manifest

This manifest is the authoritative source of which notebooks are supported release
examples versus generated documentation artifacts. CI (`.github/workflows/notebooks.yml`)
executes the **supported release notebooks** list below in full; it does not hardcode
a separate, undocumented subset.

## Classification categories

- **Supported release notebook**: executed end-to-end from a clean environment using the
  installed `compitum` package; must pass with no errors. Outputs are not asserted
  byte-for-byte (timings/whitespace vary), but successful execution is required.
- **Generated wiki projection**: auto-extracted from a single fenced code block on a
  GitHub wiki page (via `scripts/migrate_wiki_code_to_notebooks.py`), one notebook per
  fence. Structurally validated only (valid notebook JSON, valid Python syntax in every
  code cell) -- most of these are excerpts of a larger page-level narrative and
  reference variables, imports, or data files established in a *different* fence on the
  same page, or external files a reader is expected to already have. That is expected
  behavior for this category, not a defect, and they are not required to execute
  standalone.
- **Historical/archive**: none currently present in this repository.
- **Data-dependent**: none of the *supported release* notebooks require external data;
  several *generated wiki projection* notebooks do (see below) -- classified under that
  category since none currently need a standalone data-dependent designation of their own.
- **Intentionally non-executable documentation**: none currently present (every notebook
  in this repository contains at least one runnable code cell).

## Supported release notebooks (13) -- executed in CI

| Notebook | Notes |
|---|---|
| `notebooks/Getting_Started.ipynb` | |
| `notebooks/Examples_Tour.ipynb` | |
| `notebooks/Integration_Snippets.ipynb` | |
| `notebooks/CLI_Routing_From_Prompt.ipynb` | Fixed this pass: a corrupted, over-escaped shell/jq reference cell made the whole file invalid JSON. Rewritten with `jq --arg` (avoids fragile nested-quote interpolation). |
| `notebooks/examples/demo_route.ipynb` | Fixed this pass: needed CWD pinned to repo root (nbmake sets CWD to the notebook's own directory) so the CLI subprocess it shells out to can find `configs/*.yaml`; `raise SystemExit(main())` also needed to stop raising on a *successful* (0) exit, which nbclient otherwise reports as a cell failure. |
| `notebooks/examples/batch_route_demo.ipynb` | Same `sys.argv`-pollution + `SystemExit(0)` fixes as above (Jupyter's own kernel-launch args leak into `sys.argv`, colliding with the script's own argparse). |
| `notebooks/examples/certificate_card.ipynb` | Same fixes, plus supplies the required `--prompt` argument explicitly (was `required=True` in the original CLI). |
| `notebooks/examples/synth_bench.ipynb` | Same `sys.argv`/`SystemExit(0)` fixes. |
| `notebooks/examples/explain_certificate_file.ipynb` | Same fixes, plus a new cell that generates a real certificate JSON via the CLI first (the script's `--input` argument is required and needs an existing file). |
| `notebooks/examples/pedagogy_control_of_error.ipynb` | Fixed this pass: needed the repo-root CWD pin (reads `configs/*.yaml` by relative path). |
| `notebooks/examples/Fusion_Quickstart.ipynb` | Fixed this pass: `kernelspec.name` was `"compitum"`, a kernel that doesn't exist in a standard environment (changed to `"python3"`); needed the repo-root CWD pin; its `!python script.py ...` shell-magic cells were replaced with `subprocess.run([sys.executable, ...], check=True)` -- `!python` was silently resolving to a different interpreter without `compitum`/`pandas` installed under nbmake's kernel, causing downstream `FileNotFoundError`s on files that were never actually produced. |
| `notebooks/examples/Supercon_Quickstart.ipynb` | Same three fixes as Fusion_Quickstart (parallel notebook, same underlying pattern). |
| `notebooks/examples/bridge_demo.ipynb` | Already passing; no changes needed. |

## Generated wiki projections (55) -- structurally validated only

Location: `notebooks/wiki_snippets/**/*.ipynb`. Regenerable via
`python scripts/migrate_wiki_code_to_notebooks.py` from `compitum.wiki/*.md` (a separate,
locally-cloned wiki checkout, not part of this repository's tracked tree).

Validation performed this pass: every one of the 55 files parses as valid notebook JSON,
and every code cell parses as syntactically valid Python (`ast.parse`, allowing for
IPython-magic lines like `!pip install ...` which are valid only inside a real kernel,
not under plain `ast.parse` -- verified those specific cases directly via `nbmake`
instead of flagging them as false-positive syntax errors).

Two real defects were found and fixed this pass (not context-dependent -- genuine bugs):

- `Glossary/auto_glossary_1.ipynb`: markdown-escaped underscores (`free\_energy`,
  `beta\_t`, ...) had leaked into the extracted Python source from the wiki page's own
  (incorrect) escaping, making it invalid Python. Also called a `free_energy()` function
  that has never existed in `compitum.energy` (the real API is the `SymbolicFreeEnergy`
  class). Rewritten to compute the same utility formula directly, matching
  `docs/Certificate-Schema.md`'s documented formula, and now executes successfully.
- `Home/auto_home_1.ipynb`: initially flagged by a naive `ast.parse`-based check due to
  `!pip install` / `!compitum route` shell-magic lines, which are invalid *plain* Python
  syntax but valid IPython syntax. Confirmed via direct `nbmake` execution that this file
  already passes for real -- not a defect, a limitation of the syntax-only pre-check.

27 of 55 fail full execution when run standalone via `nbmake` (28 pass). Spot-checked a
representative sample across every failing family and confirmed each failure is a
context-dependent fragment, not a bug:

- Referencing a variable/import established in a *different* code fence on the same wiki
  page that isn't part of this specific fence (e.g. `Control-of-Error-in-Practice`'s 5
  fences excerpt annotated pseudocode like `is_boundary = (gap < threshold ...)` for
  illustration, not as standalone-runnable code; `Math-Constraints-Duality-and-Shadow-Prices`'s
  fences 2-5 use `np`/helper functions defined only in fence 1 of the same page).
  Families affected: `Control-of-Error-in-Practice` (1-5), `Cookbook` (1, 2, 4-6, 9-12),
  `Design-Notes` (1), `Math-Constraints-Duality-and-Shadow-Prices` (2-5),
  `Math-Kernel-Density-and-Coherence` (2-3), `Math-SPD-Metrics-and-Riemannian-Geometry`
  (2-3), `Math-Trust-Regions-and-Optimization` (3), `Output-Format` (1),
  `Trace-Glossary` (1).
- Referencing an external data file the reader is expected to have generated themselves
  from a prior step shown elsewhere on the page (e.g. `Cookbook/auto_cookbook_1.ipynb`
  globs `batch_results/trace_*.json`; `FAQ/auto_faq_1.ipynb` reads a literal `trace.json`).

None of these 27 are treated as release-blocking; they are excerpts by construction, not
independently-executable examples, and CI does not execute them.

## CI wiring

`.github/workflows/notebooks.yml`'s `execute` job runs the 13 supported release notebooks
above via `pytest --nbmake`. A separate, lighter job validates every `wiki_snippets`
notebook is parseable (valid JSON) without requiring successful execution.
