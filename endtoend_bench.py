#!/usr/bin/env python3
"""
endtoend_bench.py -- run end-to-end LP benchmarks (solve to convergence)
                     with per-tolerance sub-sweeps.

Companion to bench.py (fixed-work / iteration-cap workflow). Where bench.py
pins every tolerance to 1e-30 and lets the ITER_LIMIT decide when to stop,
this script sets a real convergence tolerance (default: 1e-4 and 1e-6 in
sequence) and imposes only a wall-clock cap (EE_TIME_LIMIT_S, 3600s).

Key differences vs bench.py:
  - The tolerance is the *only* stopping criterion the user cares about;
    every run either converges (Optimal / Solved), hits the wall-clock cap
    (Time Limit), or crashes.
  - Runs land in runs_endtoend/<tol>/<solver>/<stem>__N<n>/ so multiple
    tolerances don't overwrite each other. bench.py's runs/ is untouched.
  - Writes results_endtoend_tol<tol>.csv (not results.csv).
  - Sweeps multiple tolerances into separate subdirs so 1e-4 and 1e-6
    can be requested in one go: `python endtoend_bench.py --tol 1e-4 1e-6`.

Shared with bench.py:
  - The instance grid (bench.INSTANCES).
  - Log parsing (_cuopt_parse, _dpdlp_parse) and log-reuse cache.
  - Worker-coordination scheme (atomic .claims/ files) so N end-to-end
    workers on N nodes cooperate without a scheduler.
  - Solver env / argv builders (LD_LIBRARY_PATH, mpirun wiring, etc.).

Instances marked in PRESOLVE_THESE are replaced by their Gurobi-presolved
`.mps` counterpart (see gurobi_things/presolve.py) so we skip the solver's
internal presolver on problems where PSLP / Papilo crashes with
std::length_error or OOMs.

Usage:
    # default: sweep both 1e-4 and 1e-6 sequentially, all 3 solvers
    python endtoend_bench.py

    # only 1e-6, all solvers
    python endtoend_bench.py --tol 1e-6

    # only cuopt-distributed at 1e-4
    python endtoend_bench.py --tol 1e-4 --solver cuopt-distributed

    python endtoend_build_csv.py --tol 1e-4     # -> results_endtoend_tol1e-4.csv
    python plot_endtoend.py --tol 1e-4          # -> plots/endtoend_tol1e-4*.png
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import bench  # re-uses INSTANCES, SOLVERS, Solver, _stem_of, run_one, etc.
from bench import (
    INSTANCES,
    Solver,
    _stem_of,
    run_one,
    _cuopt_parse,
    _dpdlp_parse,
)


HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# End-to-end config
# ---------------------------------------------------------------------------
# All end-to-end runs land under runs_endtoend/<tol>/<solver>/<stem>__N<n>.
# The tolerance sits ABOVE the solver in the path so consolidating a CSV per
# tolerance is a straight `find runs_endtoend/<tol>/`.
EE_RUNS_DIR = HERE / "runs_endtoend"

# Wall-clock cap. Match D-PDLP's typical "large-scale benchmark" cap and cuopt's
# same value; larger caps are trivially added on the command line.
EE_TIME_LIMIT_S = 3600

# GPU counts per solver for the end-to-end sweep. Kept tiny on purpose:
#   - cuopt-distributed / dpdlp are meant to run at N=8 (multi-GPU point);
#   - cuopt-base is a single-GPU baseline (no dist wrapper).
# bench.py's separate "strong scaling" set (N=1,2,4,8) is orthogonal.
EE_GPU_COUNTS: dict[str, list[int]] = {
    "cuopt-distributed": [8],
    "dpdlp":             [8],
    "cuopt-base":        [1],
}

# Which solvers to actually run under --solver all. cuopt-base is included by
# request so we get a 1-GPU baseline in every end-to-end sweep.
EE_SOLVERS: dict[str, Solver] = {
    "cuopt-distributed": bench.SOLVERS["cuopt-distributed"],
    "dpdlp":             bench.SOLVERS["dpdlp"],
    "cuopt-base":        bench.SOLVERS["cuopt-base"],
}


def ee_gpu_counts_for(solver: Solver) -> list[int]:
    """Number of GPUs to sweep for this solver in an end-to-end run."""
    return EE_GPU_COUNTS.get(solver.name, [8])


# ---------------------------------------------------------------------------
# Presolve substitution
# ---------------------------------------------------------------------------
#   A few of our instances make the solver's built-in presolver (Papilo/PSLP)
# crash or OOM (std::length_error on int32-overflow, or hours of preprocessing
# for a small gain). For those, we pre-generate a Gurobi-presolved MPS via
# gurobi_things/presolve.py and tell the end-to-end sweep to use THAT file
# INSTEAD of the raw one.
#
# Rules of the road:
#   (1) An instance stem listed in PRESOLVE_THESE below is REPLACED in the
#       sweep grid by its gurobi-presolved variant -- the raw MPS is not
#       run at all for that stem. Only one variant per stem lands in the
#       CSV / plot, keeping the report uncluttered.
#   (2) The substituted path is passed to the solver with presolve
#       DISABLED (cuopt: --presolve 0, D-PDLP: --no_presolve), so
#       presolve_s / setup_s in the CSV reflect the raw pre-iter cost
#       without Papilo/PSLP work.
#   (3) The stem must exactly match Path(instance).stem stripped of the
#       .mps[.gz|.bz2|.lz4] suffixes -- same convention as bench._stem_of.
# ---------------------------------------------------------------------------

_GUROBI_PRESOLVED_DIR = "/home/scratch.vmostovoi_gpu/gurobi_things/gurobi_presolved"

# Stems (NOT full paths) of bench.INSTANCES entries whose raw MPS should
# be REPLACED with the gurobi-presolved variant at sweep time. The
# resolver:
#   raw path .../<stem>.mps       ->   <_GUROBI_PRESOLVED_DIR>/<stem>_gurobi_presolved.mps
# Add a stem here after you've produced the presolved MPS.
PRESOLVE_THESE: set[str] = {
    "psr_100",
    "C5_bigger_sanitized",
    "ELMOD_876_10_noVEnames",
    "design_match",
    "qap-tho-150",
}


def _gurobi_presolved_path_for(stem: str) -> str:
    """Return the canonical gurobi-presolved MPS path for a raw stem."""
    return f"{_GUROBI_PRESOLVED_DIR}/{stem}_gurobi_presolved.mps"


def _maybe_substitute_presolved(instance_path: str) -> str:
    """If Path(instance_path).stem is in PRESOLVE_THESE, return the
    corresponding gurobi-presolved file path; otherwise return the input
    path unchanged."""
    stem = _stem_of(instance_path)
    if stem in PRESOLVE_THESE:
        return _gurobi_presolved_path_for(stem)
    return instance_path


def _is_already_presolved(instance: str) -> bool:
    """True if this instance path lives under the gurobi_presolved
    directory (== was produced by an external presolver and must NOT be
    re-presolved by the solver)."""
    return instance.startswith(_GUROBI_PRESOLVED_DIR + "/")


def _all_instances() -> list[str]:
    """The end-to-end sweep grid = bench.INSTANCES with every stem
    listed in PRESOLVE_THESE swapped for its gurobi-presolved variant.
    The raw MPS for those stems is NOT run.

    Warns (to stderr) once per missing gurobi-presolved file so a typo
    or a not-yet-produced variant fails loudly instead of silently
    falling back to the raw path.
    """
    out: list[str] = []
    for raw in INSTANCES:
        subbed = _maybe_substitute_presolved(raw)
        if subbed is not raw and not Path(subbed).is_file():
            print(f"!! PRESOLVE_THESE lists stem {_stem_of(raw)!r} but the "
                  f"expected file {subbed} does not exist. "
                  f"Run gurobi_things/presolve.py on {raw} first.",
                  file=sys.stderr)
        out.append(subbed)
    return out


# ---------------------------------------------------------------------------
# Per-tolerance solver params
# ---------------------------------------------------------------------------
# Tolerance is passed on the CLI (--tol 1e-4 / 1e-6). We turn that into
# solver-specific flags at run time. Module-scoped so _cuopt_params /
# _dpdlp_params can read it (they don't take a tol argument; the same
# solver structs are shared with bench.py).
_EE_TOL: str = "1e-4"


def _cuopt_tol_pairs() -> list[tuple[str, str]]:
    """cuopt PDLP tolerances at _EE_TOL. Only the *relative* knobs are set:
    the absolute-* tolerances were dropped on purpose so that we don't
    accidentally over-constrain the termination criterion for problems
    with tiny objective magnitude (an abs-1e-6 target is unreachable for
    obj values around 1e-2, and just wastes iterations). Matches D-PDLP's
    default which is also relative-only (`--eps_opt`/`--eps_feas`)."""
    tol = _EE_TOL
    return [
        ("relative_primal_tolerance",  tol),
        ("relative_dual_tolerance",    tol),
        ("relative_gap_tolerance",     tol),
    ]


def _ee_cuopt_params(already_presolved: bool = False) -> list[tuple[str, str]]:
    """cuopt distributed PDLP end-to-end. presolve=1 by default; flip
    to 0 when the MPS was already produced by an external presolver
    (i.e. the instance path lives under _GUROBI_PRESOLVED_DIR).

    Uses `--mps-reader experimental-fast`, the SIMD MPS parser now
    wired into the distributed loader path. Earlier builds rejected
    the flag with `Unknown argument: --mps-reader`; if you point at an
    older cuopt_cli, remove this line or the run will refuse to start.
    """
    presolve = "0" if already_presolved else "1"
    return [
        ("use-distributed-pdlp",       "true"),
        ("method",                     "1"),
        ("presolve",                   presolve),
        ("mps_reader",                 "experimental-fast"),
        ("time_limit",                 str(EE_TIME_LIMIT_S)),
        ("iteration_limit",            "1000000000"),
        *_cuopt_tol_pairs(),
        ("log_to_console",             "true"),
    ]


def _ee_cuopt_basic_params(already_presolved: bool = False) -> list[tuple[str, str]]:
    """Single-GPU cuopt (no `--use-distributed-pdlp`, no MPI). Same
    tolerances / mps reader / presolve toggle as the distributed one."""
    presolve = "0" if already_presolved else "1"
    return [
        ("method",                     "1"),
        ("presolve",                   presolve),
        ("mps_reader",                 "experimental-fast"),
        ("time_limit",                 str(EE_TIME_LIMIT_S)),
        ("iteration_limit",            "1000000000"),
        *_cuopt_tol_pairs(),
        ("log_to_console",             "true"),
    ]


def _ee_dpdlp_flags(already_presolved: bool = False) -> list[tuple[str, str]]:
    """D-PDLP end-to-end. eps_opt / eps_feas both set to _EE_TOL.

    `no_presolve` is passed as a boolean flag when the MPS was already
    reduced by Gurobi. eps_infeas_detect is intentionally left off in the
    end-to-end sweep -- unlike the fixed-work bench, we WANT D-PDLP to
    detect a primal-infeasible problem and terminate rather than burn
    the entire wall-clock cap on hopeless iterations."""
    tol = _EE_TOL
    flags = [
        ("iter_limit",                   "1000000000"),
        ("time_limit",                   str(EE_TIME_LIMIT_S)),
        ("eps_opt",                      tol),
        ("eps_feas",                     tol),
    ]
    return flags


def _ee_dpdlp_bool_flags(already_presolved: bool = False) -> list[str]:
    flags = ["verbose"]
    if already_presolved:
        flags.append("no_presolve")
    return flags


# ---------------------------------------------------------------------------
# End-to-end runners
# ---------------------------------------------------------------------------
def _tol_runs_dir(tol: str) -> Path:
    """runs_endtoend/<tol>/ -- root of a per-tolerance sweep."""
    return EE_RUNS_DIR / tol


def _log_path_for(solver: Solver, instance: str, n_gpus: int, tol: str) -> Path:
    """Where the per-run log lands. Directory layout parallels bench.py's
    runs/, but the top level is per-tolerance so 1e-4 and 1e-6 don't fight."""
    root = _tol_runs_dir(tol) / solver.name
    root.mkdir(parents=True, exist_ok=True)
    if solver.name == "dpdlp":
        # D-PDLP writes summary.txt in a per-run directory; keep the same
        # here so parsing is unchanged.
        run_dir = root / f"{_stem_of(instance)}__N{n_gpus}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "run.log"
    # cuopt / cuopt-basic just need a flat file per run.
    return root / f"{_stem_of(instance)}__N{n_gpus}.log"


def _run_dir_for(solver: Solver, instance: str, n_gpus: int, tol: str) -> Path:
    """Companion run directory. Matches solver.run_dir() shape."""
    root = _tol_runs_dir(tol) / solver.name
    return root / f"{_stem_of(instance)}__N{n_gpus}"


def _claim_path(solver: Solver, instance: str, n_gpus: int, tol: str) -> Path:
    """Atomic .claims/ file (one per (solver, instance, n_gpus)) that lets
    N end-to-end workers on N nodes cooperate. Same convention as bench."""
    claims_dir = _tol_runs_dir(tol) / ".claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    return claims_dir / f"{solver.name}__{_stem_of(instance)}__N{n_gpus}.claim"


def _try_claim(claim_path: Path, stale_after_s: int = 300) -> bool:
    """Try to atomically create the claim file. Returns True on success.
    A stale claim (mtime older than stale_after_s) is stolen so a dead
    worker doesn't strand work forever."""
    now = time.time()
    try:
        # O_EXCL: fail if it already exists.
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{os.getpid()}@{time.time():.0f}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            age = now - claim_path.stat().st_mtime
        except FileNotFoundError:
            return False
        if age >= stale_after_s:
            # Steal by truncating + rewriting.
            try:
                with claim_path.open("w") as f:
                    f.write(f"{os.getpid()}@{time.time():.0f}\n")
                return True
            except OSError:
                return False
        return False


def _try_reuse_existing_log(solver: Solver, instance: str, n_gpus: int,
                            tol: str) -> dict | None:
    """If a finished log for this (solver, instance, n_gpus, tol) already
    exists on disk, parse it and return the metrics dict; otherwise None.
    Delegates to bench._try_reuse_existing_log so the parsing rules stay
    in one place. bench.py's version takes (solver, instance, n_gpus)
    and looks under bench.RUNS_DIR; here we temporarily monkey-patch that
    to our tolerance-specific dir."""
    saved_runs_dir = bench.RUNS_DIR
    bench.RUNS_DIR = _tol_runs_dir(tol)
    try:
        return bench._try_reuse_existing_log(solver, instance, n_gpus)
    finally:
        bench.RUNS_DIR = saved_runs_dir


def _run_one(solver: Solver, instance: str, n_gpus: int, tol: str) -> dict:
    """Actually launch the solver. Returns the parsed metrics dict."""
    log_path = _log_path_for(solver, instance, n_gpus, tol)
    run_dir = _run_dir_for(solver, instance, n_gpus, tol)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch bench's RUNS_DIR for the duration of the run so the
    # solver's own log_path/run_dir builders write in the correct
    # tolerance subdir. This keeps bench.py's Solver wiring reusable.
    saved_runs_dir = bench.RUNS_DIR
    bench.RUNS_DIR = _tol_runs_dir(tol)
    saved_params = None
    saved_flags = None
    saved_bool_flags = None
    already = _is_already_presolved(instance)
    try:
        # Build argv/env with tolerance + presolve flags overridden.
        if solver.name == "cuopt-distributed":
            saved_params = bench.CUOPT_PARAMS
            bench.CUOPT_PARAMS = _ee_cuopt_params(already_presolved=already)
        elif solver.name == "cuopt-base":
            saved_params = bench.CUOPT_BASIC_PARAMS
            bench.CUOPT_BASIC_PARAMS = _ee_cuopt_basic_params(already_presolved=already)
        elif solver.name == "dpdlp":
            saved_flags = bench.DPDLP_FLAGS
            saved_bool_flags = bench.DPDLP_BOOL_FLAGS
            bench.DPDLP_FLAGS = _ee_dpdlp_flags(already_presolved=already)
            bench.DPDLP_BOOL_FLAGS = _ee_dpdlp_bool_flags(already_presolved=already)

        _lp, metrics, rc = run_one(solver, instance, n_gpus)
        metrics["exit_code"] = rc
        return metrics
    finally:
        bench.RUNS_DIR = saved_runs_dir
        if saved_params is not None and solver.name == "cuopt-distributed":
            bench.CUOPT_PARAMS = saved_params
        if saved_params is not None and solver.name == "cuopt-base":
            bench.CUOPT_BASIC_PARAMS = saved_params
        if saved_flags is not None:
            bench.DPDLP_FLAGS = saved_flags
        if saved_bool_flags is not None:
            bench.DPDLP_BOOL_FLAGS = saved_bool_flags


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--solver",
        choices=("cuopt-distributed", "cuopt-base", "dpdlp", "all"),
        default="all",
        help="which solver to run. 'all' = cuopt-distributed + cuopt-base + dpdlp."
    )
    p.add_argument(
        "--tol",
        nargs="+",
        choices=("1e-4", "1e-6"),
        default=["1e-4", "1e-6"],
        metavar="TOL",
        help="convergence tolerance(s) for every solver "
             "(cuopt relative primal/dual/gap; D-PDLP eps_opt/eps_feas). "
             "Accepts one or more of 1e-4 / 1e-6; the sweeps run sequentially in "
             "the order given. Runs at different tolerances land in separate "
             "runs_endtoend/<tol>/ subdirs so they don't overwrite each other. "
             "Default: '1e-4 1e-6' (1e-4 first, then 1e-6).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print what would run without executing anything",
    )
    p.add_argument(
        "--stale-claim-after", type=int, default=300, metavar="SEC",
        help="reclaim a peer worker's claim after this many seconds of "
             "silence (default 300).",
    )
    return p.parse_args()


def _run_sweep_for_tol(args: argparse.Namespace, tol: str) -> int:
    """One end-to-end sweep at the given tolerance. Extracted so main()
    can call it once per --tol value with clean state."""
    global _EE_TOL
    _EE_TOL = tol

    if args.solver == "all":
        active = [EE_SOLVERS["cuopt-distributed"],
                  EE_SOLVERS["dpdlp"],
                  EE_SOLVERS["cuopt-base"]]
    else:
        active = [EE_SOLVERS[args.solver]]

    grid_instances = _all_instances()
    n_presolved = sum(1 for i in grid_instances if _is_already_presolved(i))
    n_runs = sum(len(ee_gpu_counts_for(s)) for s in active) * len(grid_instances)

    print(f"=== endtoend_bench (tol={tol}): {n_runs} runs "
          f"({len(active)} solver(s) x {len(grid_instances)} instances; "
          f"N per solver: "
          f"{[(s.name, ee_gpu_counts_for(s)) for s in active]}) ===")
    print(f"    runs dir     : {_tol_runs_dir(tol)}")
    print(f"    time_limit   : {EE_TIME_LIMIT_S}s per run")
    print(f"    presolve     : cuopt=1, dpdlp=on for raw MPS; "
          f"disabled for {n_presolved} externally-presolved instance(s)")
    if PRESOLVE_THESE:
        subs = [
            f"{stem} -> {_gurobi_presolved_path_for(stem)}"
            for stem in sorted(PRESOLVE_THESE)
        ]
        print(f"    presolved sub: {len(PRESOLVE_THESE)} stem(s) substituted "
              f"via PRESOLVE_THESE (raw MPS NOT run for these):")
        for line in subs:
            print(f"        {line}")
    print("(workers only produce per-run logs; "
          f"consolidate later with `python endtoend_build_csv.py --tol {tol}`)")

    if args.dry_run:
        for solver in active:
            for instance in grid_instances:
                for n in ee_gpu_counts_for(solver):
                    if _try_reuse_existing_log(solver, instance, n, tol) is not None:
                        flag = "reuse"
                    else:
                        cp = _claim_path(solver, instance, n, tol)
                        try:
                            age = time.time() - cp.stat().st_mtime
                            flag = f"CLAIMED(age={age:.0f}s)" if age < args.stale_claim_after else "STALE"
                        except FileNotFoundError:
                            flag = "TODO"
                    print(f"  [{flag}] {solver.name:20s} {_stem_of(instance):40s} N={n}")
        return 0

    # Real run: iterate grid, claim, run, log a short summary.
    rc = 0
    counter = 0
    for solver in active:
        for instance in grid_instances:
            for n in ee_gpu_counts_for(solver):
                counter += 1
                stem = _stem_of(instance)
                prefix = f"[{counter:2d}/{n_runs}] {solver.name} {stem} N={n}"

                reused = _try_reuse_existing_log(solver, instance, n, tol)
                if reused is not None:
                    # bench._try_reuse_existing_log returns (log_path, parsed, exit_code)
                    _lp, r_metrics, _rc = reused
                    print(f"{prefix} ... reused (status={r_metrics.get('status','?')} "
                          f"iter={r_metrics.get('iterations','?')} "
                          f"total={r_metrics.get('total_s','?')}s)")
                    continue

                claim = _claim_path(solver, instance, n, tol)
                if not _try_claim(claim, stale_after_s=args.stale_claim_after):
                    print(f"{prefix} ... skipped (claimed by peer worker)")
                    continue

                try:
                    print(f"{prefix} ...", end=" ", flush=True)
                    metrics = _run_one(solver, instance, n, tol)
                    print(f"status={metrics.get('status','?'):>10s} "
                          f"iters={metrics.get('iterations','?'):>8} "
                          f"total={metrics.get('total_s','?')}s "
                          f"gpu_peak={metrics.get('gpu_peak_mb','?')}MB "
                          f"peak2={metrics.get('gpu_peak2_mb','?')}MB "
                          f"[exit={metrics.get('exit_code','?')}]")
                except Exception as e:
                    print(f"!! {solver.name} {stem} N={n} crashed: {e}")
                    rc = rc or 1
                finally:
                    try:
                        claim.unlink()
                    except OSError:
                        pass
    return rc


def main() -> int:
    """Entry point. Loops through --tol values in order; returns the
    first non-zero rc from any per-tol sweep, so if a tighter tolerance
    aborts we still see the earlier looser-tolerance results on disk."""
    args = parse_cli()

    tolerances = list(args.tol)
    print(f"### endtoend_bench: sweeping tolerance(s) in order: "
          f"{' -> '.join(tolerances)}")

    rc_final = 0
    for i, tol in enumerate(tolerances, start=1):
        if len(tolerances) > 1:
            print(f"\n{'#' * 20} sweep {i}/{len(tolerances)}: tol={tol} "
                  f"{'#' * 20}")
        rc = _run_sweep_for_tol(args, tol)
        rc_final = rc_final or rc
    return rc_final


if __name__ == "__main__":
    sys.exit(main())
