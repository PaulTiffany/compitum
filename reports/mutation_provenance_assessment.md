# Mutation Provenance Assessment — Pre-Release Gate

**Question:** Was the mutmut/Cosmic Ray certification documented in `MUTATION_HARDENING_STATUS.md`
and `reports/mutation_summary.json` a genuinely clean, single reproduction against commit
`fe995f8`?

**Verdict: No.** A fresh clean-tree mutation-certification run is required and was performed
(see `reports/mutation_clean_verification.json` and the updated `reports/mutation_summary.json`
for its results).

## Assessment against the six required conditions

| # | Condition | Met? | Evidence |
|---|---|---|---|
| 1 | Fresh mutation databases, work directories, caches | **No** | `artifacts/mutation_cache/*.mutmut-cache` files predate this release pass by days/weeks, generated across many separate local sessions. No single fresh `.mutmut-cache` existed for a run against `fe995f8` before this gate. |
| 2 | Clean environment or clean checkout | **No** | All prior mutmut/Cosmic Ray work ran against whatever local working-tree state existed at the time (not an isolated worktree pinned to one commit), across many different commits as the branch evolved. |
| 3 | Real tests executed by both mutation engines | **Partial** | mutmut was genuinely run for every file at some point. Cosmic Ray was only run for a handful of files, historically, and mostly predates this release's own fixes by months (see `RELEASE_PROGRAM.md`'s forensic recovery of `session.sqlite`, `cr_full.sqlite` — both from Oct/Nov 2025). |
| 4 | No results inherited from prior campaigns | **No** | `mutation_summary.json`'s own `verified_by` field is explicit about this: entries read `"local"`, `"CI (2026-07-20)"`, `"CI (2026-07-20 and 2026-07-21, twice)"` — i.e., results were deliberately carried forward and accumulated across a multi-day campaign, not reset. |
| 5 | Every artifact explicitly identifies commit `fe995f8` | **No** | No prior mutation artifact references `fe995f8` — that commit did not exist until this release pass's final evidence commit. All prior mutation evidence is tied to earlier commits in the branch's history. |
| 6 | No production-source changes after the campaign began | **No** (moot) | Since no single coherent "campaign" against `fe995f8` exists, this condition cannot be evaluated as satisfied — production source changed continuously throughout the multi-day accumulation the existing evidence represents. |

## Conclusion

None of the six conditions were satisfiable from existing evidence. Per the explicit instruction,
a final clean mutation-certification run was performed against the frozen `fe995f8` tree —
see `reports/mutation_clean_verification.json` for the fresh, from-scratch results (isolated
worktree, cleared caches, single run, every artifact tagged with commit `fe995f8`).

This assessment is verification-only. It intentionally does not re-litigate or re-fix anything
already resolved in `MUTATION_HARDENING_STATUS.md` — it exists solely to establish whether the
frozen release commit's actual behavior, verified fresh and from scratch, matches what was
believed.
