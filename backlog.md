# Backlog

## In progress

## Waiting on me

- `polish-2026-10-30` is on `main` through a PR with green CI (CI has never run on GitHub), by merge commit or fast-forward, never squash or rebase.
- `v0.1.0` is tagged after the merge, with `CITATION.cff` and the changelog date matching the tag.
- O1 is decided: bundle the 2.1 MB of StRD files or keep fetching them.
- The Claude co-author trailers on the 22 existing commits are either kept or rewritten (14 are on public `main`; rewriting the other 8 changes `73eb634`, which `results/full/` cites).

## Decided

- 2026-10-30: merges keep commit `73eb634` in public history (no squash, no rebase).
- 2026-10-23: StRD files are fetched and checksum-verified, not committed, until O1 is decided.

## Proposed

- The study has run on a second platform (Linux, another BLAS), so the LAPACK-dependent cells carry a measured platform spread.
- Part B has a secondary outcome that canonicalises exchangeable parameters (Lanczos, Gauss, ENSO), so equivalent minima count as correct.
- GPU paths are measured on a CUDA machine (DESIGN 3.2).
- CI has a job that runs the tests on the declared lower bounds.
