#!/usr/bin/env python3
"""
bench.py - LP solver benchmark *worker*.

This script runs one or more LP solvers on every (instance, n_gpus) pair
and writes one run-log per tuple under runs/. It does NOT write results.csv.

To turn the runs/ directory into a CSV, run `python build_csv.py` once,
on whichever host you want the consolidated view. That separation is
deliberate: it lets you start as many `bench.py` workers as you like, on
any number of hosts sharing runs/, with zero concurrent-CSV-writer race
conditions.

Three solvers are wired in:
  - "cuopt-distributed" : NVIDIA cuopt_cli with distributed PDLP (1D
                          bipartite partition, KaMinPar by default).
  - "cuopt-base"        : same binary as cuopt-distributed but
                          use_distributed_pdlp=false (single-GPU baseline).
                          Only sweeps N=1.
  - "dpdlp"             : D-PDLP cupdlpx-dist (2D grid partition,
                          MPI-launched). Grid size is left to auto-detect.

Resume-safe / idempotent: the on-disk run log is the source of truth.

  - If the canonical log file for (solver, instance, n_gpus) already
    exists and ended cleanly (has the '### exit_code=' footer), the
    solver is NOT re-invoked. The reuse path is intra-bench only; the
    CSV is rebuilt independently by build_csv.py.
  - To force a re-run, delete the log file.

Log layout (all under runs/):
  runs/cuopt-distributed/<stem>__N<n>.log
  runs/cuopt-base/<stem>__N<n>.log
  runs/dpdlp/<stem>__N<n>/run.log

Every run log carries a '### key=value' header (pre-run) and footer
(post-run) block: solver, instance, n_gpus, exit_code, gpu_peak_mb,
gpu_peak2_mb, per_gpu_peak_mb, started, finished, argv. The footer is
written immediately after the subprocess returns, so a partial log
(no footer) marks an interrupted run and triggers a re-attempt next time.

Multi-host parallelism (no configuration required):

Launch `python bench.py` on as many machines as you like, as long as
they all see the same scaling_bench/ directory (shared scratch / NFS).
Workers coordinate via atomic claim files under runs/.claims/; each
tuple is claimed exclusively by exactly one worker, with stale-claim
takeover after a few minutes' silence so a dead worker doesn't strand
work. No partitioning flags, no scheduler -- just start more workers.

Stdlib only.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

try:
    import pynvml  # type: ignore
except ImportError:
    pynvml = None


# ===========================================================================
# config -- edit these
# ===========================================================================

HERE        = Path(__file__).resolve().parent
RUNS_DIR    = HERE / "runs"

# How often the GPU memory sampler probes NVML, in milliseconds.
# NVML calls cost ~0.1-1ms each, so 10ms = 100 Hz is comfortably within the
# overhead budget even with 8 GPUs (per-loop cost ~1-8ms). Drop lower (e.g.
# 2-5ms) only if you suspect sub-10ms transient spikes you're missing;
# below ~2ms the polling thread starts contending for a CPU core and may
# perturb the very thing you're measuring.
MEM_SAMPLE_PERIOD_MS = 10

# --- cuopt ----------------------------------------------------------------
# cuopt_cli build. Both CUOPT_PARAMS (distributed) and CUOPT_BASIC_PARAMS
# (single-GPU baseline) point at this binary. Note:
#   - The build must support `--mps-reader experimental-fast` (the SIMD
#     MPS parser wired into the distributed loader path) and
#     `--distributed-pdlp-partitioner kaminpar` (the cherry-picked
#     multi-threaded partitioner). The mostovoi_cuopt tree has kaminpar
#     since the 505c54bd cherry-pick; the experimental-fast reader
#     landed shortly after. If your CUOPT_CLI doesn't support both,
#     runs will exit with a `ValidationError` from the cli; check with
#     `cuopt_cli --help`.
#   - The bullshit_mostovoi_cuopt build also adds explicit "Setup time:"
#     and "Step time:" log lines from our patch. mostovoi_cuopt lacks those;
#     bench.py falls back to deriving setup_s from the iter-table's first
#     row in either case.
CUOPT_CLI          = Path("/home/scratch.vmostovoi_gpu/mostovoi_cuopt/cpp/build/cuopt_cli")
# Conda env cuopt_cli was built against (CUDA 13.3). bench.py uses this
# absolute path instead of $CONDA_PREFIX, so the bench works regardless of
# which env the user invoked it from -- cuopt and dpdlp have incompatible
# CUDA/MPI stacks and must each use their own toolchain.
CUOPT_CONDA_PREFIX = Path("/home/scratch.vmostovoi_gpu/.conda/envs/cuopt_dev_133")

# Fixed-work design: ITER_LIMIT PDLP iters with every tolerance pinned to
# 1e-30 so the solver never short-circuits on convergence and every
# run does identical work. Wall-clock is therefore directly comparable
# across (instance, n_gpus). Single source of truth: all three solver
# configs (cuopt distributed, cuopt base, D-PDLP) reference this string
# verbatim, so bumping the iter cap is a one-line change here.
ITER_LIMIT: str = "20000"

CUOPT_PARAMS: list[tuple[str, str]] = [
    ("use_distributed_pdlp",       "true"),    # required: enable distributed PDLP path
    ("distributed_pdlp_partitioner", "kaminpar"),  # auto|dummy|metis|kaminpar (multi-threaded => much faster than METIS on big graphs)
    ("mps_reader",                 "experimental-fast"),   # new SIMD MPS parser, distributed-path-safe (replaces plain "fast" which used to crash on the distributed loader on some inputs)
    ("method",                     "1"),       # PDLP only (no concurrent dual simplex)
    ("presolve",                   "0"),       # None - distributed PDLP rejects anything else
    ("iteration_limit",            ITER_LIMIT),  # fixed iter cap = fixed work
    ("time_limit",                 "1e9"),     # effectively infinite; iteration_limit is the cap
    ("absolute_primal_tolerance",  "1e-30"),
    ("relative_primal_tolerance",  "1e-30"),
    ("absolute_dual_tolerance",    "1e-30"),
    ("relative_dual_tolerance",    "1e-30"),
    ("absolute_gap_tolerance",     "1e-30"),
    ("relative_gap_tolerance",     "1e-30"),
    ("log_to_console",             "true"),
]


# --- cuopt (single-GPU baseline, no distributed PDLP) ---------------------
# Same binary, same fixed-work strategy, but use_distributed_pdlp=false so
# the solver follows the legacy single-shard PDLP path. Useful as the "this
# is what one GPU does without any of the distributed framework overhead"
# baseline against which both distributed cuopt and D-PDLP are compared.
# Only N=1 is meaningful here (the distributed path is what enables N>1).
CUOPT_BASIC_PARAMS: list[tuple[str, str]] = [
    ("use_distributed_pdlp",       "false"),   # the baseline knob
    ("mps_reader",                 "experimental-fast"),   # SIMD MPS parser, same as distributed cuopt
    ("method",                     "1"),
    ("presolve",                   "0"),
    ("iteration_limit",            ITER_LIMIT),
    ("time_limit",                 "1e9"),
    ("absolute_primal_tolerance",  "1e-30"),
    ("relative_primal_tolerance",  "1e-30"),
    ("absolute_dual_tolerance",    "1e-30"),
    ("relative_dual_tolerance",    "1e-30"),
    ("absolute_gap_tolerance",     "1e-30"),
    ("relative_gap_tolerance",     "1e-30"),
    ("log_to_console",             "true"),
]


# --- D-PDLP ---------------------------------------------------------------
DPDLP_BIN          = Path("/home/scratch.vmostovoi_gpu/D-PDLP/build/cupdlpx-dist")
# Conda env D-PDLP was built against. Provides libcudart.so.12, libmpi.so.40,
# libnccl.so.2, libcusparse.so.12, libcublas.so.12, and the mpirun launcher.
# We don't depend on the user activating this env -- we point at it
# explicitly so cuopt and D-PDLP can each use their own toolchain.
DPDLP_CONDA_PREFIX = Path("/home/scratch.vmostovoi_gpu/.conda/envs/dpdlp")
DPDLP_MPIRUN       = DPDLP_CONDA_PREFIX / "bin" / "mpirun"

# Where OpenMPI/PMIx should put its session dirs. Default /tmp on our compute
# nodes is a small tmpfs that gets full quickly (a single N=8 run leaves
# dozens of Unix-socket files under /tmp/ompi.<pid>/), which then causes
# every subsequent mpirun on the same node to die at startup with:
#     PMIx detected a call to mkdir was unable to create the desired directory:
#       Directory: /tmp/ompi.4347   Error: No space left on device
# and exit_code=213. Point OMPI at a per-node subdir on scratch (roomy, and
# unique per node so parallel workers don't stomp each other). OpenMPI still
# adds its own uid/jobid suffix inside, so cleanup remains automatic.
DPDLP_TMPDIR_BASE  = Path("/home/scratch.vmostovoi_gpu/tmp_dpdlp")
# Grid policy: leave grid_size unset and D-PDLP picks (their README default).
# Iter cap + tight tolerances match cuopt's fixed-work strategy.
DPDLP_FLAGS: list[tuple[str, str]] = [
    ("iter_limit",        ITER_LIMIT),
    ("time_limit",        "1e9"),
    ("eps_opt",           "1e-30"),
    ("eps_feas",          "1e-30"),
    # Prevent infeasibility-detection shortcut: D-PDLP otherwise terminates
    # very early on hard problems (e.g. psr_100 N=8) with "Primal Infeasible",
    # which produces nonsensical step_s = ~0 in the iter loop and breaks the
    # fixed-work comparison. 1e-30 makes the check effectively unreachable.
    ("eps_infeas_detect", "1e-30"),
    # NOTE: termination-evaluation frequency (D-PDLP's `--eval_freq`, default
    # 200) is intentionally left at its default to stay apples-to-apples with
    # cuopt, whose analogous `major_iteration` (Stable3 preset default = 200)
    # is hardcoded in solve.cu and not CLI-exposed. Both solvers therefore
    # run with major-iteration = 200 inside the ITER_LIMIT-iter window.
]
# Boolean (no-value) flags emitted as bare --name.
DPDLP_BOOL_FLAGS: list[str] = [
    "verbose",       # required for the iter table (setup_s extraction)
    "no_presolve",   # match cuopt's --presolve 0
]


INSTANCES = [
    "/home/scratch.vmostovoi_gpu/mostovoi_cuopt/datasets/linear_programming/afiro_original.mps",
    "/home/scratch.cmaes_sw/zib03.mps",
    
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/psr_100.mps",                      # 54G
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/C5_bigger_sanitized.mps",          # 34G
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/C5_baseline_sanitized.mps",        # 14G
    "/home/scratch.bbozkaya_gpu/datasets/GAMS/ELMOD_876_10_noVEnames.mps",         
    "/home/scratch.bbozkaya_gpu/datasets/GAMS/VERYLARGE/BEAM_4032_11_8_CLI.mps",    # 132G (Burcin VERYLARGE)


    # design_match raw (123G) is too big for PSLP presolve (SIGSEGV in transpose)
    # -> substituted at sweep time by design_match_gurobi_presolved.mps via
    #    endtoend_bench.PRESOLVE_THESE. Keep the raw path listed so the
    #    substitution key matches; the raw file itself is not opened.
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/design_match.mps",                 # 123G (presolved)
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/heat-source-easy.mps",      # 5.6G
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/heat-source-hard.mps",      # 5.6G
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/mediterranean-shipping.mps",# 27G
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/production-imventory.mps",  # 14G
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/qap-tho-150.mps",           # 48G
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/qap-wil-100.mps",           # 11G
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/supply-chain.mps",          # 32G
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/tsp-gaia-10m.mps",          # 25G
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/world-shipping.mps",        
    #"/home/scratch.vmostovoi_gpu/datasets/big_lp/tsp-gaia-100m.mps",                # 397G

    # H. Mittelmann LPfeas test set (plato.asu.edu/ftp/lptestset/), .mps
    # extracted from *.mps.bz2. Sizes are modest compared to the Google /
    # GAMS monsters so PSLP presolve should handle them raw.
    "/home/scratch.vmostovoi_gpu/datasets/mittleman_mps/Dual2_5000.mps",             # 30M rows x 33M cols x 93M nnz
    "/home/scratch.vmostovoi_gpu/datasets/mittleman_mps/dlr2.mps",

    # Open-energy benchmark (zen-garden), see github.com/ZEN-universe/ZEN-garden
    "/home/scratch.vmostovoi_gpu/datasets/zen-garden-eur-PI-28-200ts.mps",           # 2.9G

    # Amazon LP-relaxation instances. Amazon's LP/ folder actually contains
    # MIPs (99% binary vars) which they told us to relax to [0,1] and treat
    # as continuous. Produced by gurobi_things/npz_to_mps.py --relax.
    "/home/scratch.vmostovoi_gpu/datasets/amazon_lp/amazon_lp003.mps",               # 6.9G, 17M vars x 1M rows x 129M nnz
    "/home/scratch.vmostovoi_gpu/datasets/amazon_lp/amazon_lp004.mps",               # 6.9G, 17M vars x 1M rows x 129M nnz

    # Multicommodity-flow benchmark instances from Oliver Hinder's
    # `large-scale-LP-test-problems` repo (github.com/ohinder/...),
    # pre-generated and shared by cmaes_sw. Same suite reported in
    # the D-PDLP paper (arXiv 2601.07628) Table 4 as `mcf_<C>_<W>_<S>`
    # where C=commodities, W=warehouses, S=stores.
    "/home/scratch.cmaes_sw/large-scale-LP-test-problems/large-problem-instances/multicommodity-flow-instance_2500_100_500.mps.gz",  # 3.0G gz, 1.5M rows x 126M cols x 254M nnz
    "/home/scratch.cmaes_sw/large-scale-LP-test-problems/large-problem-instances/multicommodity-flow-instance_5000_100_250.mps",     # 21G,     1.8M rows x 127.5M cols x 257.5M nnz
    "/home/scratch.cmaes_sw/large-scale-LP-test-problems/large-problem-instances/multicommodity-flow-instance_5000_50_500.mps",      # 20G,     2.8M rows x 126M cols x 254M nnz

    # PSR6 model_de_<N>_scenarios stochastic power-system LPs, shared by
    # bbozkaya on Jul 21 2026. These are the same instances that
    # appear in his internal cuPDLP-vs-mPDLP vs Xpress Barrier table:
    # (rows, cols, nnz reported by bbozkaya's Gurobi presolve log)
    #   20-scen  : 28.0M rows x 36.1M cols x  93.3M nnz  (presolved: 27.7M / 31.9M / 88.4M)
    #   50-scen  : 70.1M rows x 90.1M cols x 233.1M nnz  (presolved: 69.3M / 79.7M / 220.9M)
    #  100-scen  : 140M   rows x 180M   cols x 466M   nnz (presolved: 138.5M / 159.4M / 441.7M)
    # All three are well under the 1B-nnz zone where cuopt starts to
    # hit int32-overflow crashes (see design_match investigation), so
    # they should go through PSLP + solve without special handling.
    # Loaded raw (mps.gz supported by cuopt's experimental-fast parser
    # and by D-PDLP's MPS reader); we do NOT switch to bbozkaya's
    # `20_presolved.lp` because (a) it exists only for 20-scen, and
    # (b) .lp isn't supported by D-PDLP's loader.
    "/home/scratch.bbozkaya_gpu/datasets/PSR6/model_de_20_scenarios.mps.gz",         # 705M gz -> 93M nnz
    "/home/scratch.bbozkaya_gpu/datasets/PSR6/model_de_50_scenarios.mps.gz",         # 1.8G gz -> 233M nnz
    "/home/scratch.bbozkaya_gpu/datasets/PSR6/model_de_100_scenarios.mps.gz",        # 3.6G gz -> 466M nnz
]
N_GPUS_LIST = [1, 2, 4, 8]


# ===========================================================================
# Solver abstraction
#
# Each solver provides four small pieces of behaviour:
#   1. binary             : Path to executable.
#   2. build_argv(...)    : returns the full argv list for one run.
#   3. build_env(...)     : returns the env dict (LD_LIBRARY_PATH, etc.).
#   4. parse_log(...)     : returns a dict of timing/iteration metrics.
#   5. log_path(...)      : where to write the run's stdout/stderr.
#   6. run_dir(...)       : where the run may write auxiliary files (D-PDLP
#                           writes its summary.txt there). cuopt has no such
#                           directory; it returns RUNS_DIR.
# ===========================================================================


@dataclass(frozen=True)
class Solver:
    name: str
    binary: Path
    build_argv:   Callable[[str, int, "Solver"], list[str]]
    build_env:    Callable[[dict, int, "Solver"], dict]
    parse_log:    Callable[[Path, Path], dict]
    log_path:     Callable[[str, int], Path]
    run_dir:      Callable[[str, int], Path]


def gpu_counts_for(solver: Solver) -> list[int]:
    """Per-solver GPU sweep list. Single-GPU baselines (cuopt-base) only run
    at N=1 because they don't use the distributed code path; all other
    solvers use the global N_GPUS_LIST."""
    if solver.name == "cuopt-base":
        return [1]
    return list(N_GPUS_LIST)


def _stem_of(path: str) -> str:
    """Strip .mps[.gz|.bz2|.lz4] from a basename."""
    name = Path(path).name
    for ext in (".gz", ".bz2", ".lz4"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    if name.endswith(".mps"):
        name = name[: -4]
    return name


# NOTE: bench.py no longer reads $CONDA_PREFIX. Each solver references the
# absolute path of the conda env it was BUILT against (CUOPT_CONDA_PREFIX /
# DPDLP_CONDA_PREFIX above). This makes the bench safe to run from any env
# -- or even from no env at all -- and prevents the cuopt-vs-dpdlp CUDA/MPI
# ABI mismatch we used to hit when LD_LIBRARY_PATH leaked the wrong libs in.


# ---------------------------------------------------------------------------
# cuopt adapter
# ---------------------------------------------------------------------------

def _cuopt_flag_args() -> list[str]:
    """CUOPT_PARAMS as a flat argv list of --kebab-case + value."""
    out: list[str] = []
    for k, v in CUOPT_PARAMS:
        out += ["--" + k.replace("_", "-"), v]
    return out


def _cuopt_argv(instance: str, n_gpus: int, solver: Solver) -> list[str]:
    return [str(solver.binary), *_cuopt_flag_args(), instance]


def _cuopt_env(base_env: dict, n_gpus: int, solver: Solver) -> dict:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n_gpus))
    # cuopt_cli LD_LIBRARY_PATH: binary's build dir FIRST (so its libcuopt.so
    # wins over any copy installed into the conda env), then cuopt env's lib
    # (CUDA 13.3 stack), then METIS for the legacy partitioner. The cuopt
    # env path is hardcoded, NOT inherited from the active env -- see the
    # comment above about CUDA ABI separation.
    _extra_ld = [
        str(solver.binary.parent),
        str(CUOPT_CONDA_PREFIX / "lib"),
        "/home/scratch.vmostovoi_gpu/metis64/lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(_extra_ld + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    return env


def _cuopt_log_path(instance: str, n_gpus: int) -> Path:
    return RUNS_DIR / "cuopt-distributed" / f"{_stem_of(instance)}__N{n_gpus}.log"


def _cuopt_run_dir(instance: str, n_gpus: int) -> Path:
    # cuopt writes nothing to disk except its log; the run dir is the same as
    # the log's parent. build_csv.py passes this to parse_log() but the
    # cuopt parser ignores it (only dpdlp uses run_dir to find summary.txt).
    return RUNS_DIR / "cuopt-distributed"


# --- cuopt_basic adapter (reuses cuopt's env / parser; only argv & paths differ) ---

def _cuopt_basic_flag_args() -> list[str]:
    out: list[str] = []
    for k, v in CUOPT_BASIC_PARAMS:
        out += ["--" + k.replace("_", "-"), v]
    return out


def _cuopt_basic_argv(instance: str, n_gpus: int, solver: Solver) -> list[str]:
    return [str(solver.binary), *_cuopt_basic_flag_args(), instance]


def _cuopt_basic_log_path(instance: str, n_gpus: int) -> Path:
    # Separate subdir so we don't collide with distributed cuopt's
    # runs/<stem>__N1.log file.
    return RUNS_DIR / "cuopt-base" / f"{_stem_of(instance)}__N{n_gpus}.log"


def _cuopt_basic_run_dir(instance: str, n_gpus: int) -> Path:
    return RUNS_DIR / "cuopt-base"


# Log-line regexes for cuopt:
#   METIS    partitioned bipartite graph: nvtx=... nb_parts=4 edge_cut=... in 0.821s
#   KaMinPar partitioned bipartite graph (attempt 1/3, seed=42): nvtx=... in 0.123s
#   Setup time: 1.823s (before PDLP iteration loop)   [bullshit_mostovoi_cuopt only]
#   Step time:  21.989s   Time/step: 4.398ms   ...    [bullshit_mostovoi_cuopt only]
#   Status: Iteration Limit   ...   Iterations: 20000   Time: 23.812s
#
# Fallback for builds without Setup/Step lines: extract elapsed time from
# the PDLP iter-table. First row's elapsed time ~= setup cost.
RX_CUOPT_PART = re.compile(
    r"(?P<engine>METIS|KaMinPar)\s+partitioned bipartite graph.*?"
    r"in\s+(?P<t>\d+(?:\.\d+)?)s"
)
RX_CUOPT_SETUP  = re.compile(r"^Setup time:\s+(?P<t>\d+(?:\.\d+)?)s")
RX_CUOPT_STEP   = re.compile(r"^Step time:\s+(?P<t>\d+(?:\.\d+)?)s")
# Present when cuopt reaches either its legacy Papilo presolver or the
# current default PSLP presolver. Lines look like:
#     Papilo presolve time: 133.69s
#     PSLP presolve time: 52.25s
# The value is wall-clock seconds spent inside presolve, and it is INCLUDED
# in the final "Status: ... Time: N.NNs" wall clock reported at end of
# run, so we do NOT need to add it to total_s again -- we only capture
# it for reporting parity with D-PDLP.
RX_CUOPT_PRESOLVE = re.compile(
    r"^(?:Papilo|PSLP) presolve time:\s+(?P<t>\d+(?:\.\d+)?)s"
)
RX_CUOPT_STATUS = re.compile(
    r"^Status:\s+(?P<status>.+?)\s+Objective:.+?"
    r"Iterations:\s+(?P<it>\d+)\s+Time:\s+(?P<t>\d+(?:\.\d+)?)s"
)
RX_CUOPT_TABLE_HDR = re.compile(
    r"^\s*Iter\s+Primal Obj|^distributed_pdlp:\s+shard build done\s+in"
)
RX_CUOPT_TABLE_ROW = re.compile(
    r"^\s*(?P<it>\d+)\s+"
    r"[+-]?\d[\d.eE+\-]*\s+[+-]?\d[\d.eE+\-]*\s+[+-]?\d[\d.eE+\-]*\s+"
    r"[+-]?\d[\d.eE+\-]*\s+[+-]?\d[\d.eE+\-]*\s+"
    r"(?P<t>\d+(?:\.\d+)?)s?\s*$"
)


def _cuopt_parse(log_path: Path, run_dir: Path) -> dict:
    out = dict(total_s=None, partition_s=None, partition_engine=None,
               setup_s=None, presolve_s=None, step_s=None,
               iterations=None, status=None)
    # We accept any line matching RX_CUOPT_TABLE_ROW as an iter row --
    # historically we gated on RX_CUOPT_TABLE_HDR ("Iter Primal Obj" /
    # "distributed_pdlp: shard build done in") but some builds don't
    # emit either header, so on a crashed-at-time-limit run (SIGSEGV
    # on teardown, no "Status:" line) we'd end up with an empty row.
    # The row regex has enough numeric structure (iter + 5 floats +
    # time-in-seconds) that noise-matching a non-iter line is a
    # non-issue in practice.
    first_row_elapsed = None
    last_row_elapsed  = None
    last_row_it       = None
    with log_path.open(errors="replace") as f:
        for line in f:
            if (m := RX_CUOPT_PART.search(line)):
                out["partition_s"]      = float(m["t"])
                out["partition_engine"] = m["engine"]
                continue
            if (m := RX_CUOPT_SETUP.match(line)):
                out["setup_s"] = float(m["t"])
                continue
            if (m := RX_CUOPT_STEP.match(line)):
                out["step_s"] = float(m["t"])
                continue
            if (m := RX_CUOPT_PRESOLVE.match(line)):
                out["presolve_s"] = float(m["t"])
                continue
            if (m := RX_CUOPT_STATUS.match(line)):
                out["status"]     = m["status"].strip()
                out["iterations"] = int(m["it"])
                out["total_s"]    = float(m["t"])
                continue
            if (m := RX_CUOPT_TABLE_ROW.match(line)):
                t = float(m["t"])
                if first_row_elapsed is None:
                    first_row_elapsed = t
                last_row_elapsed = t
                last_row_it      = int(m["it"])

    # Fallback #1: no "Status:" line means the process was killed before
    # PDLP could print its final summary. This is the shape of a run that
    # crashed on teardown after hitting the wall-clock cap (SIGSEGV /
    # SIGKILL on shutdown), or was terminated externally. We still have
    # the last iter row, which is a fine proxy for what actually
    # happened. Populate total_s / iterations from it, and label the run
    # TIME_LIMIT so the plot classifies it correctly (not as "no data
    # crash") -- if we didn't get anywhere near the cap it stays
    # CRASHED_MID_RUN. Threshold = 90% of the endtoend cap
    # (EE_TIME_LIMIT_S in endtoend_bench.py, 3600s by default; hardcoded
    # here to keep bench.py agnostic of the endtoend harness).
    if out["total_s"] is None and last_row_elapsed is not None:
        out["total_s"]    = last_row_elapsed
        out["iterations"] = last_row_it
        if out["status"] is None:
            out["status"] = "TIME_LIMIT" if last_row_elapsed >= 0.9 * 3600.0 else "CRASHED_MID_RUN"

    if out["setup_s"] is None and first_row_elapsed is not None:
        # The current cuopt build's PDLP clock starts after MPS loading:
        # logs can show MPS read=86.98s followed by iter-0 elapsed=84.235s.
        # Therefore iter-0 is already post-MPS setup and needs no subtraction.
        out["setup_s"] = first_row_elapsed
    # step_s = actual iter-loop time as reported in the trace. Preferred
    # source: last_row_elapsed - first_row_elapsed (matches what the user
    # sees when they read the iter table directly). Fallback: total - setup,
    # which can over-report by several seconds because PDLP does final
    # convergence/KKT work between the last iter row and the "Status:" line
    # (e.g. ~9s on psr_100 N=8: iter 20000 at 147.6s, Status at 156.8s).
    if (out["step_s"] is None
            and first_row_elapsed is not None
            and last_row_elapsed is not None
            and last_row_elapsed > first_row_elapsed):
        out["step_s"] = last_row_elapsed - first_row_elapsed
    if out["step_s"] is None and out["total_s"] is not None and out["setup_s"] is not None:
        out["step_s"] = max(0.0, out["total_s"] - out["setup_s"])
    return out


# ---------------------------------------------------------------------------
# D-PDLP adapter
#
# Launcher: `mpirun -n N <bin> <MPS> <OUTPUT_DIR> --verbose --iter_limit 20000
#           --time_limit 1e9 --eps_opt 1e-30 --eps_feas 1e-30 --no_presolve`.
# Output : one machine-readable <inst>_summary.txt in OUTPUT_DIR with
#          `Runtime (sec): ...` (= solve-loop time), `Iterations Count: ...`,
#          `Termination Reason: ...` (= status). Cleaner than scraping
#          stdout.
# stdout : separate permutation and data-distribution timers. Their sum is
#          setup_s; this remains reliable when NFS corruption loses iter 0.
# ---------------------------------------------------------------------------

def _dpdlp_argv(instance: str, n_gpus: int, solver: Solver) -> list[str]:
    rd = solver.run_dir(instance, n_gpus)
    rd.mkdir(parents=True, exist_ok=True)
    # Absolute path to dpdlp env's mpirun -- avoids depending on $PATH and
    # in particular lets bench.py run from cuopt_dev_133 (no MPI installed).
    argv: list[str] = [str(DPDLP_MPIRUN), "-n", str(n_gpus),
                       str(solver.binary), instance, str(rd)]
    for k, v in DPDLP_FLAGS:
        argv += [f"--{k}", v]
    for k in DPDLP_BOOL_FLAGS:
        argv += [f"--{k}"]
    return argv


def _dpdlp_env(base_env: dict, n_gpus: int, solver: Solver) -> dict:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n_gpus))
    # D-PDLP needs its own libcupdlpx-dist.so + dpdlp env's libnccl /
    # libmpi / libcudart (CUDA 12 stack). No metis64 dependency.
    # Bin dir first so its sibling .so wins.
    _extra_ld = [
        str(solver.binary.parent),
        str(DPDLP_CONDA_PREFIX / "lib"),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(_extra_ld + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    # Also expose dpdlp env's bin so mpirun's companion launchers
    # (orted/hydra_pmi_proxy/...) are picked up alongside mpirun itself.
    env["PATH"] = ":".join([str(DPDLP_CONDA_PREFIX / "bin"), env.get("PATH", "")])
    # Reroute OMPI/PMIx's session dirs off /tmp (which is a tiny tmpfs that
    # fills up and kills every subsequent mpirun on the same node with
    # exit_code=213 "No space left on device" -- see DPDLP_TMPDIR_BASE
    # comment). Per-host subdir keeps parallel workers isolated. OMPI adds
    # its own uid/jobid path components inside, so cleanup is automatic.
    host = socket.gethostname().split(".")[0] or "unknown"
    tmp = DPDLP_TMPDIR_BASE / host
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"]                    = str(tmp)
    env["OMPI_MCA_orte_tmpdir_base"] = str(tmp)
    env["PMIX_SERVER_TMPDIR"]        = str(tmp)
    return env


def _dpdlp_log_path(instance: str, n_gpus: int) -> Path:
    # D-PDLP writes auxiliary files (summary.txt, solution files); put both
    # log + outputs under one per-run directory.
    rd = RUNS_DIR / "dpdlp" / f"{_stem_of(instance)}__N{n_gpus}"
    return rd / "run.log"


def _dpdlp_run_dir(instance: str, n_gpus: int) -> Path:
    return RUNS_DIR / "dpdlp" / f"{_stem_of(instance)}__N{n_gpus}"


# stdout: `[Timer] Permuting LP Problem took 16.512 seconds.`
RX_DPDLP_TIMER_PERMUTE = re.compile(
    r"\[Timer\]\s+Permuting LP Problem\s+took\s+(?P<t>\d+(?:\.\d+)?)\s+seconds"
)
# Covers both Bcast -> Partition and Partition -> P2P Send variants.
RX_DPDLP_TIMER_PART = re.compile(
    r"\[Timer\]\s+Data Distribution.*?took\s+(?P<t>\d+(?:\.\d+)?)\s+seconds"
)
# Iter row from utils.cu:483:
#   printf("%6d %.1e | %8.1e  %8.1e | %.1e %.1e %.1e | %.1e %.1e %.1e \n", ...)
# i.e. iter, time, primal_obj, dual_obj, |, abs_pres, abs_dres, gap, |, rel_pres, rel_dres, rel_gap.
RX_DPDLP_ITER_ROW = re.compile(
    r"^\s*(?P<it>\d+)\s+"
    r"(?P<elapsed>\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s+\|\s+"
    r"[+-]?\d[\d.eE+\-]*\s+[+-]?\d[\d.eE+\-]*\s+\|"
)


def _dpdlp_parse(log_path: Path, run_dir: Path) -> dict:
    out = dict(total_s=None, partition_s=None, partition_engine=None,
               setup_s=None, presolve_s=None, step_s=None,
               iterations=None, status=None)

    # Primary source of truth: D-PDLP writes <stem>_summary.txt in run_dir.
    # Format is one "Key: value" per line. Find the file (one per run).
    summary = None
    if run_dir.is_dir():
        candidates = sorted(run_dir.glob("*_summary.txt"))
        if candidates:
            summary = candidates[0]
    # D-PDLP's "Runtime (sec)" is JUST the solve loop; its presolve time
    # lives on a separate line ("Presolve Time (sec)"). cuopt's reported
    # "Status: ... Time: N.NNs" already folds presolve into that single
    # number, so for parity we need to sum them here and expose the
    # end-to-end wall clock as total_s. We keep the raw solve-only value
    # in step_s (below) so the phase breakdown is not lost.
    solve_s: float | None = None
    if summary is not None and summary.is_file():
        for line in summary.read_text(errors="replace").splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            try:
                if k == "Runtime (sec)":
                    solve_s = float(v)
                elif k == "Presolve Time (sec)":
                    out["presolve_s"] = float(v)
                elif k == "Iterations Count":
                    out["iterations"] = int(v)
                elif k == "Termination Reason":
                    out["status"] = v
            except ValueError:
                pass
    # D-PDLP's Runtime excludes both pre-loop timers. MPS parsing remains
    # excluded because D-PDLP does not expose a loader timer.
    permute_s: float | None = None
    partition_s: float | None = None
    if log_path.is_file():
        with log_path.open(errors="replace") as f:
            for line in f:
                if (m := RX_DPDLP_TIMER_PERMUTE.search(line)):
                    permute_s = float(m["t"])
                    continue
                if (m := RX_DPDLP_TIMER_PART.search(line)):
                    partition_s = float(m["t"])
                    out["partition_s"]      = partition_s
                    out["partition_engine"] = "2D-grid"

    if permute_s is not None or partition_s is not None:
        out["setup_s"] = (permute_s or 0.0) + (partition_s or 0.0)

    if solve_s is not None:
        out["step_s"] = solve_s
        out["total_s"] = (
            solve_s
            + (out.get("presolve_s") or 0.0)
            + (out.get("setup_s") or 0.0)
        )
    return out


# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

SOLVERS: dict[str, Solver] = {
    "cuopt-distributed": Solver(
        name="cuopt-distributed", binary=CUOPT_CLI,
        build_argv=_cuopt_argv, build_env=_cuopt_env, parse_log=_cuopt_parse,
        log_path=_cuopt_log_path, run_dir=_cuopt_run_dir,
    ),
    # Single-GPU baseline: same binary, same parser, just use_distributed_pdlp=false.
    # Restricted to N=1 by gpu_counts_for() because the distributed path is what
    # enables N>1.
    "cuopt-base": Solver(
        name="cuopt-base", binary=CUOPT_CLI,
        build_argv=_cuopt_basic_argv, build_env=_cuopt_env, parse_log=_cuopt_parse,
        log_path=_cuopt_basic_log_path, run_dir=_cuopt_basic_run_dir,
    ),
    "dpdlp": Solver(
        name="dpdlp", binary=DPDLP_BIN,
        build_argv=_dpdlp_argv, build_env=_dpdlp_env, parse_log=_dpdlp_parse,
        log_path=_dpdlp_log_path, run_dir=_dpdlp_run_dir,
    ),
}


def _solver_prereq_issues(solver: Solver) -> list[str]:
    """Return a list of human-readable problems that would prevent this
    solver from running (missing binary, missing mpirun, missing conda env
    libs, etc.). Empty list = good to go. Reported once at sweep startup so
    we can skip a misconfigured solver cleanly instead of crashing partway
    through the run list."""
    issues: list[str] = []
    if not solver.binary.is_file():
        issues.append(f"binary not found: {solver.binary}")

    if solver.name in ("cuopt-distributed", "cuopt-base"):
        lib = CUOPT_CONDA_PREFIX / "lib"
        if not lib.is_dir():
            issues.append(f"cuopt env lib dir missing: {lib} "
                          f"(set CUOPT_CONDA_PREFIX in bench.py)")
    elif solver.name == "dpdlp":
        if not DPDLP_MPIRUN.is_file():
            issues.append(f"mpirun not found at {DPDLP_MPIRUN} "
                          f"(set DPDLP_CONDA_PREFIX in bench.py, or "
                          f"`mamba install -n dpdlp openmpi` if the env exists)")
        lib = DPDLP_CONDA_PREFIX / "lib"
        if not lib.is_dir():
            issues.append(f"dpdlp env lib dir missing: {lib} "
                          f"(set DPDLP_CONDA_PREFIX in bench.py)")
    return issues


# ===========================================================================
# GPU memory sampler
#
# Polls visible GPUs via NVML every `period_ms` ms while the child solver
# runs. Records the per-GPU max-`used` across the entire run.
#   peak_mb   : max across visible GPUs (the OOM-relevant number).
#   peak2_mb  : second-largest across visible GPUs. Compare against peak_mb
#               to diagnose master-GPU concentration:
#                 peak2 ~= peak   -> load balanced across shards
#                 peak2 <<  peak  -> one card (usually rank 0 / "master")
#                                    carries disproportionate state. cuopt's
#                                    distributed PDLP gathers the full
#                                    primal/dual/reduced_cost onto rank 0 at
#                                    solution-return time, so we expect this
#                                    signature on large instances.
#
# Raw `nvmlDeviceGetMemoryInfo(h).used` -- NO baseline subtraction. This
# is intentional: we want the absolute occupancy on each card, including
# the CUDA driver context, so the result is directly comparable to each
# card's hardware limit (e.g. ~180 GiB on a B200).
# ===========================================================================


class MemSampler:
    """Background-thread peak-memory sampler. Context manager.

        with MemSampler(visible_gpus=[0,1,2,3]) as ms:
            subprocess.run(...)
        print(ms.peak_mb)            # max across visible GPUs
        print(ms.peak2_mb)           # 2nd-largest (= 0 for single-GPU runs)
        print(ms.per_gpu_peak_mb)    # per-GPU peaks, in NVML index order
    """

    def __init__(self, visible_gpus: list[int], period_ms: int = 50):
        if pynvml is None:
            sys.exit(
                "pynvml not installed; cannot sample GPU memory. "
                "Run `pip install pynvml` (or `conda install -c conda-forge pynvml`) "
                "in your active env."
            )
        self.visible_gpus = list(visible_gpus)
        self.period_s     = period_ms / 1000.0
        self._stop        = threading.Event()
        self._thread: threading.Thread | None = None
        self._per_gpu_peak_mb = [0.0] * len(self.visible_gpus)

    def __enter__(self) -> "MemSampler":
        pynvml.nvmlInit()
        # CUDA_VISIBLE_DEVICES remaps device indices but pynvml still uses
        # physical indices. We pass in the *physical* indices we expect the
        # solver to see (e.g. [0,1,2,3] when n_gpus=4 and the env sets
        # CUDA_VISIBLE_DEVICES=0,1,2,3).
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.visible_gpus]
        self._thread = threading.Thread(target=self._loop, name="MemSampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            for i, h in enumerate(self._handles):
                try:
                    used_mb = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)
                    if used_mb > self._per_gpu_peak_mb[i]:
                        self._per_gpu_peak_mb[i] = used_mb
                except Exception:
                    # Don't kill the bench if NVML hiccups (e.g. driver
                    # ABI mismatch on a single sample). Keep going.
                    pass
            # wait() returns early if .set() was called -> fast shutdown.
            self._stop.wait(self.period_s)

    @property
    def peak_mb(self) -> float:
        """Max across visible GPUs of peak `used` memory, in MB.
        This is the OOM-relevant number (per-card hardware limit)."""
        return max(self._per_gpu_peak_mb) if self._per_gpu_peak_mb else 0.0

    @property
    def peak2_mb(self) -> float:
        """Second-largest peak across visible GPUs, in MB. Returns 0 when
        fewer than two GPUs were sampled (i.e. single-GPU runs).

        Diagnostic intent: with the distributed solvers we want every shard
        to look the same. If peak2_mb is close to peak_mb, all GPUs are
        equally loaded. If peak2_mb is much smaller, one GPU is doing
        something the others aren't -- typically the master gathering the
        full primal/dual/reduced_cost vector at solution return."""
        if len(self._per_gpu_peak_mb) < 2:
            return 0.0
        return sorted(self._per_gpu_peak_mb, reverse=True)[1]

    @property
    def per_gpu_peak_mb(self) -> list[float]:
        return list(self._per_gpu_peak_mb)


# ===========================================================================
# Multi-worker coordination via shared-filesystem atomic claims
#
# Workflow on a shared filesystem (e.g. NFS-mounted scratch):
#
#   - Each worker invokes `python bench.py` with no special flags. As soon
#     as a worker decides a (solver, instance, n_gpus) tuple needs to run,
#     it tries to atomically create the claim file
#         runs/.claims/<solver>__<stem>__N<n>.claim
#     using os.open(O_CREAT|O_EXCL). POSIX guarantees only one creator
#     wins.
#   - The losing workers see the file exists, skip the tuple, and move on
#     to the next one. They never block waiting for the holder.
#   - While the run is in flight, a background heartbeat thread `touch`es
#     the claim file every CLAIM_HEARTBEAT_S seconds so other workers can
#     tell the holder is still alive.
#   - On completion, the holder removes the claim file. The on-disk log
#     (with its '### exit_code=' footer) now suffices for the regular
#     log-as-truth reuse logic; no claim needed thereafter.
#   - If a worker dies mid-run, the claim file's mtime stops advancing.
#     Other workers consider it stale after CLAIM_STALE_AFTER_S, atomically
#     take it over, and re-run the tuple. The half-written log will be
#     overwritten by the new worker's run_one().
#
# This requires only POSIX semantics on a shared filesystem; no central
# scheduler, no daemon, no manual partitioning. Any number of workers on
# any number of machines can run concurrently against the same runs/ dir.
# ===========================================================================

CLAIMS_DIR             = RUNS_DIR / ".claims"
CLAIM_HEARTBEAT_S      = 30.0   # touch interval for live workers
CLAIM_STALE_AFTER_S    = 180.0  # > heartbeat * ~6; threshold after which
                                # a non-touching claim is presumed dead


def _claim_path(solver: Solver, instance: str, n_gpus: int) -> Path:
    return CLAIMS_DIR / f"{solver.name}__{_stem_of(instance)}__N{n_gpus}.claim"


def _write_claim_payload(claim_path: Path) -> None:
    """Identify the holder of a claim. Best-effort metadata for humans
    inspecting the .claims/ directory; the file's mere existence + mtime
    is what carries semantic weight for the coordination protocol."""
    try:
        claim_path.write_text(
            f"host={socket.gethostname()}\n"
            f"pid={os.getpid()}\n"
            f"started={datetime.now().isoformat()}\n"
        )
    except OSError:
        pass


def _try_claim(solver: Solver, instance: str, n_gpus: int) -> Path | None:
    """Atomically acquire the claim for this (solver, instance, n_gpus).
    Returns the claim path on success, None if another live worker holds
    it. Stale claims (no heartbeat for > CLAIM_STALE_AFTER_S) are taken
    over automatically.
    """
    p = _claim_path(solver, instance, n_gpus)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: nobody holds the claim yet.
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        _write_claim_payload(p)
        return p
    except FileExistsError:
        pass

    # Someone holds it -- is the heartbeat fresh?
    try:
        age = time.time() - p.stat().st_mtime
    except OSError:
        # File vanished between FileExistsError and stat -- the holder
        # just released. Retry the fast path once.
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            _write_claim_payload(p)
            return p
        except FileExistsError:
            return None
    if age <= CLAIM_STALE_AFTER_S:
        return None

    # Stale: take over. unlink + recreate exclusively. The unlink/create
    # window is technically racy (two workers might both observe a stale
    # claim and race to recreate), but only one of them will win the
    # O_EXCL recreate -- the loser correctly returns None.
    try:
        p.unlink()
    except OSError:
        pass
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        _write_claim_payload(p)
        return p
    except FileExistsError:
        return None


def _release_claim(claim_path: Path) -> None:
    try:
        claim_path.unlink()
    except OSError:
        pass


class ClaimHeartbeat:
    """Periodically refresh the mtime of a claim file so other workers
    can tell we're still alive. Used as a context manager wrapping the
    solver subprocess invocation."""

    def __init__(self, claim_path: Path, period_s: float = CLAIM_HEARTBEAT_S):
        self.claim_path = claim_path
        self.period_s   = period_s
        self._stop      = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ClaimHeartbeat":
        self._thread = threading.Thread(
            target=self._loop, name=f"ClaimHB:{self.claim_path.name}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                os.utime(self.claim_path, None)
            except OSError:
                # Claim file was removed externally -- stop touching it.
                break
            self._stop.wait(self.period_s)


# ===========================================================================
# orchestrate one run
# ===========================================================================

def run_one(solver: Solver, instance: str, n_gpus: int) -> tuple[Path, dict, int]:
    """Execute one run, unless a successful canonical log already exists.

    This second check happens after the caller acquired its claim. It protects
    successful shared-NFS logs from stale metadata reads in the earlier
    pre-claim reuse check and prevents a bad worker from truncating good work.
    Non-zero exits remain retryable and may be overwritten.
    """
    log_path = solver.log_path(instance, n_gpus)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _parse_log_metadata(log_path)
    if existing is not None and existing.get("exit_code") == 0:
        rd = solver.run_dir(instance, n_gpus)
        parsed = solver.parse_log(log_path, rd)
        parsed["gpu_peak_mb"] = existing.get("gpu_peak_mb")
        parsed["gpu_peak2_mb"] = existing.get("gpu_peak2_mb")
        return log_path, parsed, 0

    env  = solver.build_env(os.environ.copy(), n_gpus, solver)
    argv = solver.build_argv(instance, n_gpus, solver)
    rd   = solver.run_dir(instance, n_gpus)

    # We always launch with CUDA_VISIBLE_DEVICES = "0,...,n-1", so the
    # physical GPU indices to watch are exactly range(n_gpus).
    visible_gpus = list(range(n_gpus))

    t0 = datetime.now()
    with log_path.open("w") as logf:
        logf.write(f"### solver={solver.name}\n")
        logf.write(f"### instance={instance}\n")
        logf.write(f"### n_gpus={n_gpus}\n")
        logf.write(f"### CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')}\n")
        logf.write(f"### started={t0.isoformat()}\n")
        logf.write(f"### argv={' '.join(argv)}\n")
        logf.write(f"### --- solver output ---\n")
        logf.flush()
        with MemSampler(visible_gpus, period_ms=MEM_SAMPLE_PERIOD_MS) as ms:
            proc = subprocess.run(
                argv, stdout=logf, stderr=subprocess.STDOUT, env=env, check=False,
            )
        # One '### key=value' line per metadata field so build_csv.py can
        # parse them with a single regex. The per-GPU list is written too
        # (for debugging) but only peak / peak2 are propagated to the CSV.
        logf.write(f"\n### exit_code={proc.returncode}\n")
        logf.write(f"### finished={datetime.now().isoformat()}\n")
        logf.write(f"### gpu_peak_mb={ms.peak_mb:.1f}\n")
        logf.write(f"### gpu_peak2_mb={ms.peak2_mb:.1f}\n")
        logf.write(f"### per_gpu_peak_mb={ms.per_gpu_peak_mb}\n")

    parsed = solver.parse_log(log_path, rd)
    parsed["gpu_peak_mb"]  = ms.peak_mb
    parsed["gpu_peak2_mb"] = ms.peak2_mb
    return log_path, parsed, proc.returncode


# ===========================================================================
# CSV schema (produced by build_csv.py, NOT by bench.py)
# ===========================================================================

CSV_COLUMNS = [
    "solver", "instance", "n_gpus",
    # total_s is end-to-end wall time: for both cuopt and D-PDLP it now
    # includes presolve. presolve_s is the isolated Papilo / D-PDLP-presolver
    # component (None when presolve was disabled or the phase produced no
    # timing line). step_s is the pure iteration-loop portion.
    "total_s", "partition_s", "partition_engine",
    "setup_s", "presolve_s", "step_s",
    "iterations", "status",
    # gpu_peak_mb  = max across all sampled GPUs (OOM-relevant).
    # gpu_peak2_mb = 2nd-largest. Compare to peak_mb to diagnose master-GPU
    #                concentration; 0 for single-GPU runs.
    "gpu_peak_mb", "gpu_peak2_mb",
    "exit_code", "log_path",
]


# ===========================================================================
# Run logs are the source of truth.
#
# The CSV is derived from the run logs, not the other way around. Two
# direct consequences:
#
#   - Live sweep is idempotent: any (solver, instance, n_gpus) tuple whose
#     canonical log already exists and ended cleanly is reused, the CSV
#     row is filled in from the log, and the solver is NOT re-invoked.
#     This makes Ctrl-C + restart safe, makes deleting results.csv
#     non-destructive (next sweep rebuilds it), and means manual cleanups
#     don't require any special bookkeeping.
#
#   - Force a re-run by deleting the log file. Deleting just the CSV row
#     is not enough -- the next sweep would simply rebuild that row from
#     the existing log without re-executing anything.
#
# The '### key=value' header (pre-run) + footer (post-run) blocks we write
# in run_one() give us solver, instance, n_gpus, exit_code, and the GPU
# memory peaks. solver.parse_log() then pulls total_s / step_s /
# iterations / status / ... out of the solver's own output.
# ===========================================================================

# Match '### key=value' metadata lines we emit at the top and bottom of every
# run log. Keys are unquoted identifiers; values run to end-of-line.
RX_META = re.compile(r"^###\s+(?P<k>[A-Za-z_][\w]*)\s*=\s*(?P<v>.*)$")
# Embedded per_gpu_peak_mb=[...] anywhere inside a '###' line. The current
# footer puts this on its own line, but the legacy footer wrote
# '### gpu_peak_mb=X  per_gpu_peak_mb=[...]' as one line, so we have to dig
# into the value to recover it.
RX_EMBED_PER_GPU = re.compile(r"per_gpu_peak_mb\s*=\s*\[(?P<list>[^\]]*)\]")
# For parsing the per_gpu_peak_mb list ('[1234.5, 678.9, ...]'): extract every
# number, ignore brackets/commas/whitespace.
RX_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?")


def _parse_log_metadata(log_path: Path) -> dict | None:
    """Pull out the '### key=value' metadata from a run log. We write that
    metadata at both the top (pre-run: solver, instance, n_gpus, argv, ...)
    and bottom (post-run: exit_code, gpu_peak_mb, gpu_peak2_mb, ...) of the
    log, so we look at the first AND last few lines.

    Returns None when the essentials (solver, instance, n_gpus) are missing
    (e.g. file was truncated, hand-edited, or doesn't look like one of ours).
    """
    if not log_path.is_file():
        return None

    out: dict = {}
    try:
        with log_path.open(errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None

    # 30 head + 30 tail covers our metadata regardless of how big the solver's
    # iter-table is in between. Reading the whole file would also work but
    # cuopt's 20000-iter logs are several MB each.
    for line in lines[:30] + lines[-30:]:
        line = line.rstrip()
        m = RX_META.match(line)
        if not m:
            continue
        k = m["k"]
        v = m["v"].strip()
        if k == "n_gpus":
            try: out[k] = int(v)
            except ValueError: pass
        elif k == "exit_code":
            try: out[k] = int(v)
            except ValueError: pass
        elif k in ("gpu_peak_mb", "gpu_peak2_mb"):
            # Defensive split: legacy footers wrote
            # "gpu_peak_mb=12345.6  per_gpu_peak_mb=[...]" on a single line,
            # so just take the leading number.
            try: out[k] = float(v.split()[0])
            except (ValueError, IndexError): pass
        elif k == "per_gpu_peak_mb":
            out[k] = [float(x) for x in RX_NUM.findall(v)]
        else:
            out[k] = v
        # Catch a per_gpu_peak_mb=[...] embedded in the value of another key
        # (the legacy footer's gpu_peak_mb line). Only set if we haven't
        # already pulled it out via its own '###' line.
        if "per_gpu_peak_mb" not in out and (em := RX_EMBED_PER_GPU.search(line)):
            out["per_gpu_peak_mb"] = [float(x) for x in RX_NUM.findall(em["list"])]

    if "solver" not in out or "instance" not in out or "n_gpus" not in out:
        return None

    # Old logs (pre-peak2 footer) only wrote gpu_peak_mb + per_gpu_peak_mb on
    # one line. Synthesize the missing peak / peak2 from the per-GPU list.
    per = out.get("per_gpu_peak_mb")
    if isinstance(per, list) and per:
        if "gpu_peak_mb" not in out:
            out["gpu_peak_mb"] = max(per)
        if "gpu_peak2_mb" not in out:
            out["gpu_peak2_mb"] = sorted(per, reverse=True)[1] if len(per) >= 2 else 0.0

    return out


def _csv_row_for(solver_name: str,
                 instance_name: str,
                 n_gpus: int,
                 parsed: dict,
                 exit_code,
                 log_path: Path) -> list:
    """Build a CSV row (list of cells, in CSV_COLUMNS order) from the parsed
    metrics + run metadata. Lives in bench.py so build_csv.py can import it
    and produce a CSV whose schema is guaranteed in sync with the parser."""
    fmt = lambda x: "" if x is None else x
    peak_mb  = parsed.get("gpu_peak_mb")
    peak2_mb = parsed.get("gpu_peak2_mb")
    return [
        solver_name, instance_name, n_gpus,
        fmt(parsed["total_s"]), fmt(parsed["partition_s"]), fmt(parsed["partition_engine"]),
        fmt(parsed["setup_s"]), fmt(parsed.get("presolve_s")), fmt(parsed["step_s"]),
        fmt(parsed["iterations"]), fmt(parsed["status"]),
        (f"{peak_mb:.1f}"  if peak_mb  is not None else ""),
        (f"{peak2_mb:.1f}" if peak2_mb is not None else ""),
        exit_code,
        str(log_path),
    ]


def _try_reuse_existing_log(solver: Solver,
                            instance: str,
                            n_gpus: int) -> tuple[Path, dict, int | str] | None:
    """If a *finished* run log already exists at the canonical path for
    this (solver, instance, n_gpus), return (log_path, parsed_metrics,
    exit_code) so the caller can write the CSV row from it without
    re-invoking the solver. Returns None when there's no log or the log
    is partial (no '### exit_code=' footer).

    'Finished' means run_one() got to write its footer, which it does
    immediately after the subprocess returns -- so an OOM crash or a
    non-zero exit still counts as finished (we just record the failure
    and move on). To force a re-run, delete the log file.

    This shares the same parse helpers as build_csv.py, so the live
    sweep's "skip" decision and the CSV's row always agree."""
    log_path = solver.log_path(instance, n_gpus)
    meta = _parse_log_metadata(log_path)
    if meta is None or "exit_code" not in meta:
        return None
    rd = solver.run_dir(instance, n_gpus)
    parsed = solver.parse_log(log_path, rd)
    parsed["gpu_peak_mb"]  = meta.get("gpu_peak_mb")
    parsed["gpu_peak2_mb"] = meta.get("gpu_peak2_mb")
    return log_path, parsed, meta.get("exit_code", "")


# NOTE: the CSV-from-logs builder used to live here as `rescan()`. It has
# moved to `build_csv.py` to enforce a strict producer/consumer split:
#   - bench.py (any number of workers, on any hosts sharing runs/) only
#     produces per-run logs. It never touches results.csv.
#   - build_csv.py (run once, on whichever host you want the merged view)
#     walks runs/, parses every log, and writes results.csv atomically
#     from scratch.
# This avoids all multi-writer CSV races and NUL/duplicate-row corruption.


# ===========================================================================
# main: sweep
# ===========================================================================

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--solver",
        choices=("cuopt-distributed", "cuopt-base", "dpdlp", "all", "both"),
        default="all",
        help="which solver(s) to benchmark. "
             "'all' = cuopt-distributed + cuopt-base + dpdlp (default). "
             "'both' = cuopt-distributed + dpdlp (kept for back-compat).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print what would be run without executing anything",
    )
    return p.parse_args()


def main() -> int:
    args = parse_cli()

    if args.solver == "all":
        active = [SOLVERS["cuopt-distributed"], SOLVERS["cuopt-base"], SOLVERS["dpdlp"]]
    elif args.solver == "both":
        active = [SOLVERS["cuopt-distributed"], SOLVERS["dpdlp"]]
    else:
        active = [SOLVERS[args.solver]]

    if not args.dry_run:
        good: list[Solver] = []
        for s in active:
            issues = _solver_prereq_issues(s)
            if issues:
                print(f"!! skipping solver {s.name!r} -- prerequisites not met:")
                for it in issues:
                    print(f"     - {it}")
            else:
                good.append(s)
        if not good:
            sys.exit("no solvers have their prerequisites satisfied; aborting.")
        active = good

    total = sum(len(gpu_counts_for(s)) for s in active) * len(INSTANCES)
    idx   = 0
    print(f"=== bench: {total} runs ({len(active)} solver(s) x {len(INSTANCES)} instances; "
          f"N per solver: {[(s.name, gpu_counts_for(s)) for s in active]}) ===")
    print("(workers only produce per-run logs under runs/; "
          "build the merged CSV later with `python build_csv.py`.)")

    if args.dry_run:
        for solver in active:
            for instance in INSTANCES:
                for n in gpu_counts_for(solver):
                    if _try_reuse_existing_log(solver, instance, n) is not None:
                        flag = "reuse"
                    else:
                        # Peek at the claim file without acquiring it (dry-run
                        # is side-effect-free). If a live claim exists, this
                        # worker would defer; otherwise it would run.
                        cp = _claim_path(solver, instance, n)
                        try:
                            age = time.time() - cp.stat().st_mtime
                            flag = "OTHER" if age <= CLAIM_STALE_AFTER_S else "STEAL"
                        except OSError:
                            flag = "RUN  "
                    argv = solver.build_argv(instance, n, solver)
                    print(f"  [{flag}] solver={solver.name:17s} {_stem_of(instance):28s} N={n}  argv={' '.join(argv)}")
        return 0

    for solver in active:
        for instance in INSTANCES:
            stem = _stem_of(instance)
            for n in gpu_counts_for(solver):
                idx += 1
                prefix = f"  [{idx:3d}/{total}] {solver.name:17s} {stem} N={n}"

                # ---- Reuse path -----------------------------------------
                # If the canonical run log already exists and is finished
                # (has '### exit_code=' footer), do NOT re-invoke the
                # solver. CSV bookkeeping happens later in build_csv.py.
                reuse = _try_reuse_existing_log(solver, instance, n)
                if reuse is not None:
                    _, _, ec = reuse
                    print(f"{prefix} -- log exists, skip [exit={ec}]")
                    continue

                # ---- Claim path (multi-worker coordination) ------------
                # Tuple needs to be run. Try to acquire its claim. If
                # another live worker has it (on this or any other host
                # sharing runs/), skip and move on -- they'll finish it
                # and the log will appear on the shared FS. Stale claims
                # (no heartbeat for > CLAIM_STALE_AFTER_S) are taken over
                # automatically.
                claim_path = _try_claim(solver, instance, n)
                if claim_path is None:
                    print(f"{prefix} -- claimed by another worker, skip")
                    continue

                # ---- Run path -------------------------------------------
                print(f"{prefix} ...", end=" ", flush=True)
                log_path = solver.log_path(instance, n)
                p: dict = {}
                ec: int | str = -1
                launch_err: Exception | None = None
                try:
                    with ClaimHeartbeat(claim_path):
                        log_path, p, ec = run_one(solver, instance, n)
                except FileNotFoundError as e:
                    # Solver binary or mpirun moved since the startup prereq
                    # check. Record and keep going so the rest of the sweep
                    # still makes progress.
                    launch_err = e
                except KeyboardInterrupt:
                    _release_claim(claim_path)
                    raise
                except Exception as e:  # noqa: BLE001
                    launch_err = e
                finally:
                    _release_claim(claim_path)

                if launch_err is not None:
                    print(f"LAUNCH FAILED: {type(launch_err).__name__}: {launch_err}")
                    try:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with log_path.open("a") as logf:
                            logf.write(f"\n### exit_code=launch_error\n")
                            logf.write(f"### finished={datetime.now().isoformat()}\n")
                            logf.write(f"### launch_error={type(launch_err).__name__}: {launch_err}\n")
                    except OSError:
                        pass
                    continue

                fmt = lambda x: "" if x is None else x
                eng = p.get("partition_engine")
                part_label = f"{eng.lower()}={fmt(p['partition_s'])}s" if eng \
                             else f"partition={fmt(p['partition_s'])}s"
                peak_mb  = p.get("gpu_peak_mb")
                peak2_mb = p.get("gpu_peak2_mb")
                peak_label = (
                    f"gpu_peak={peak_mb:.0f}MB peak2={peak2_mb:.0f}MB"
                    if peak_mb is not None else "gpu_peak=?"
                )
                print(
                    f"total={fmt(p['total_s'])}s {part_label} "
                    f"setup={fmt(p['setup_s'])}s step={fmt(p['step_s'])}s "
                    f"{peak_label}  [exit={ec}]"
                )

    print(f"\nDone. Logs under {RUNS_DIR}. "
          f"Run `python build_csv.py` (anywhere with access to {RUNS_DIR}) "
          f"to consolidate them into results.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
