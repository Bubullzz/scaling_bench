# Archived Gurobi-presolved end-to-end runs

Historical `runs_endtoend/` artifacts for the five stems that used to be
substituted with Gurobi-presolved MPS files:

- `psr_100`
- `C5_bigger_sanitized`
- `ELMOD_876_10_noVEnames`
- `design_match`
- `qap-tho-150`

Layout mirrors `runs_endtoend/` (`<tol>/<solver>/...`).

As of 2026-08-06 the active sweep runs the **raw** MPS for these stems with
in-solver presolve disabled (`NO_PRESOLVE_THESE` in `endtoend_bench.py`).
These archived logs are kept for comparison only; they are not read by
`endtoend_build_csv.py` / plot scripts unless you point them here explicitly.
