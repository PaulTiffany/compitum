## Summary

- [ ] Purpose of this change is clear
- [ ] Tests pass locally (or explain)

## Labels to consider

- outerbench-online � run the opt-in capped online evaluation workflow
- `routerbench` — to run RouterBench job (guarded; caches dataset/models if present)
- `mutation` — to run sharded mutation (mutmut + CR quick) on changed modules

## Checklists

- [ ] CI green (lint, types, tests)
- [ ] Docs updated if needed
- [ ] Notebooks updated if relevant (executed via nbmake)

## Notes

- Heavy jobs are opt-in via labels/dispatch. See `WORKFLOWS.md` for details.


