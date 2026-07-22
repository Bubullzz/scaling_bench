#!/usr/bin/env python3
"""
scaling_bench/scripts/parse_log.py

Parse a single cuopt_cli stdout log (as written by scripts/run_sweep.sh) into a
structured dict. Stdlib only - intentionally no pandas/numpy here so this
module is importable from any Python on the target machine.

The log format we rely on is what cuopt_cli emits with `log_to_console=true`
and the rapids_logger pattern `%v` (bare message, no timestamps).

Lines we extract (all examples are the actual printf format strings used by
the binary - regexes below match them precisely):

  ### scaling_bench run header              (added by run_sweep.sh)
  ### problem=<abs path>
  ### stem=<basename>
  ### n_gpus_requested=<int>
  ### rep=<W0|W1|...|1|2|3|...>
  ### cuda_visible_devices=<csv>
  ### partition_file=<path or marker>
  ### started=<iso8601>
  ### finished=<iso8601>

  Solving a problem with <N> constraints, <M> variables (<I> integers), and <NNZ> nonzeros (distributed mps-direct path)
     Iter    Primal Obj.      Dual Obj.    Gap        Primal Res.  Dual Res.   Time
  Setup time: <t>s (before PDLP iteration loop)         (patched binary only)
   <iter> <+e> <+e>  <e>   <e>     <e>   <t>s
  PDLP finished
  Status: <termination string>   Objective: <obj>  Iterations: <N>  Time: <t>s
  Step time: <t>s   Time/step: <ms>ms   (setup <t>s of <t>s total, <pct>%)   (patched binary only)
"""

from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Optional


# ---- regex catalogue --------------------------------------------------------

# Header lines from run_sweep.sh.
_HDR = re.compile(r"^###\s+(?P<key>[A-Za-z_0-9]+)=(?P<val>.*)$")

# "Solving a problem with %d constraints, %d variables (%d integers), and %d nonzeros (distributed mps-direct path)"
_DIMS = re.compile(
    r"Solving a problem with\s+(?P<n_cstr>\d+)\s+constraints,\s+"
    r"(?P<n_vars>\d+)\s+variables\s+\((?P<n_int>\d+)\s+integers\),\s+and\s+"
    r"(?P<nnz>\d+)\s+nonzeros\s+\(distributed mps-direct path\)"
)

# Iteration table row: "%7d %+.8e %+.8e  %8.2e   %8.2e     %8.2e   %.3fs"
# The first column is an int (PDLP iteration index). Numeric columns are
# scientific. Last column ends in 's' for seconds.
_NUM = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_ITER_ROW = re.compile(
    r"^\s*(?P<iter>\d+)\s+"
    r"(?P<primal_obj>" + _NUM + r")\s+"
    r"(?P<dual_obj>" + _NUM + r")\s+"
    r"(?P<gap>" + _NUM + r")\s+"
    r"(?P<primal_res>" + _NUM + r")\s+"
    r"(?P<dual_res>" + _NUM + r")\s+"
    r"(?P<elapsed>\d+(?:\.\d+)?)s\s*$"
)

# Terminal "Status:" line. Status strings can contain spaces ("Iteration Limit",
# "Primal Infeasible", "A numerical error was encountered."), so we are
# permissive and let the next label terminate the match.
_TERMINAL = re.compile(
    r"^Status:\s+(?P<status>.+?)\s+"
    r"Objective:\s+(?P<objective>" + _NUM + r")\s+"
    r"Iterations:\s+(?P<iterations>\d+)\s+"
    r"Time:\s+(?P<total_time>\d+(?:\.\d+)?)s\s*$"
)

# "PDLP finished" sentinel emitted right before the terminal status line.
_PDLP_FINISHED = re.compile(r"^PDLP finished\s*$")

# Auto-detect line (only present when distributed_pdlp_num_gpus==-1 was
# requested, which we never do, but parse anyway for robustness).
_AUTO_DETECT = re.compile(
    r"distributed_pdlp_num_gpus == -1, auto-detected\s+(?P<n>\d+)\s+visible CUDA device"
)

# METIS timing (so we can attribute setup to partitioning specifically).
_METIS = re.compile(
    r"METIS partitioned bipartite graph:\s+nvtx=(?P<nvtx>\d+)\s+nnz=(?P<nnz>\d+)\s+"
    r"nb_parts=(?P<nb_parts>\d+)\s+edge_cut=(?P<edge_cut>-?\d+)\s+in\s+(?P<dt>\d+(?:\.\d+)?)s"
)

# Explicit "Setup time:" log line emitted by the bullshit_mostovoi_cuopt
# patch inside pdlp_solver_t::run_solver right before the main loop. Format:
#   "Setup time: %.3fs (before PDLP iteration loop)"
# This is the AUTHORITATIVE setup time when present (the binary measured it
# directly with no slop from major-iter printout cadence). For older binaries
# without the patch we fall back to first_iter_elapsed_s.
_SETUP_TIME = re.compile(
    r"^Setup time:\s+(?P<setup>\d+(?:\.\d+)?)s\b"
)

# Explicit "Step time:" log line emitted from solve_lp_distributed_from_mps
# right after the existing Status line. Format:
#   "Step time: %.3fs   Time/step: %.3fms   (setup %.3fs of %.3fs total, %.1f%%)"
_STEP_TIME = re.compile(
    r"^Step time:\s+(?P<step>\d+(?:\.\d+)?)s\s+"
    r"Time/step:\s+(?P<ms_per_step>\d+(?:\.\d+)?)ms\s+"
    r"\(setup\s+(?P<setup>\d+(?:\.\d+)?)s\s+of\s+"
    r"(?P<total>\d+(?:\.\d+)?)s\s+total,"
)

# Failure classification. Each entry is (label, compiled-regex). FIRST
# match wins, so order matters: put the most specific signatures
# (terminal status strings) before generic ones (raw CUDA errors).
#
# We deliberately leave out "Status: Iteration Limit" / "Status: Optimal"
# etc. -- those mean success and are handled by the Status terminal regex
# above; a successful run has failure_mode=None.
_FAILURE_PATTERNS = [
    # PDLP-side numerical breakdown (e.g. distributed scaling blew up).
    ("NumericalError",  re.compile(
        r"Status:\s+A numerical error was encountered",
        re.IGNORECASE,
    )),
    ("PrimalInfeasible", re.compile(
        r"Status:\s+Primal Infeasible", re.IGNORECASE,
    )),
    ("DualInfeasible",  re.compile(
        r"Status:\s+Dual Infeasible", re.IGNORECASE,
    )),
    # GPU memory / generic CUDA allocation failures. We catch the
    # canonical RMM / RAFT / std::bad_alloc strings plus rmm's
    # cuda_error_memory_allocation.
    ("OOM", re.compile(
        r"(out of memory|cudaErrorMemoryAllocation|std::bad_alloc|"
        r"RMM failure|out_of_memory|CUDA_ERROR_OUT_OF_MEMORY|"
        r"cuda(Error)?(MemoryAllocation|OutOfMemory))",
        re.IGNORECASE,
    )),
    # MPS reader / file IO failure (file too big for host RAM, etc.).
    ("MPSReaderError", re.compile(
        r"(MPS reader|failed to parse|parse error|invalid MPS|"
        r"could not open|No such file or directory)",
        re.IGNORECASE,
    )),
    # NCCL collective comms failure.
    ("NCCLError", re.compile(
        r"(NCCL error|ncclResult|ncclInternalError|ncclSystemError)",
        re.IGNORECASE,
    )),
    # cuopt's own validation guards (presolve, precision, etc.).
    ("ValidationError", re.compile(
        r"validation error", re.IGNORECASE,
    )),
    # Generic CUDA driver / OS error (driver crashed, machine sick).
    ("CUDAError", re.compile(
        r"(cudaError|cuda_error|CUDA_ERROR|cudaErrorOperatingSystem)",
        re.IGNORECASE,
    )),
    # C++ exception that wasn't categorized above.
    ("Exception", re.compile(
        r"(terminate called|what\(\):|std::runtime_error|std::exception|"
        r"^\s*Aborted|Segmentation fault|SIGSEGV|SIGABRT)",
        re.IGNORECASE,
    )),
]

# Header lines from run_sweep.sh trailing the cuopt_cli output.
_HDR_FOOTER = re.compile(r"^###\s+exit_code=(?P<ec>-?\d+)\s*$")


# ---- parser -----------------------------------------------------------------

def parse_log(path: Path) -> dict:
    """Parse one log file. Returns a flat dict; missing fields are None."""
    rec: dict = {
        "log_path": str(path),
        "problem": None,
        "stem": None,
        "n_gpus_requested": None,
        "rep": None,
        "cuda_visible_devices": None,
        "partition_file": None,
        "started": None,
        "finished": None,
        "n_cstr": None,
        "n_vars": None,
        "n_int": None,
        "nnz": None,
        "status": None,
        "objective": None,
        "iterations": None,
        "total_time_s": None,
        "first_iter": None,
        "last_iter": None,
        "first_iter_elapsed_s": None,
        "last_iter_elapsed_s": None,
        "n_iter_rows": 0,
        "metis_time_s": None,
        "metis_edge_cut": None,
        "metis_nb_parts": None,
        "auto_detected_gpus": None,
        "pdlp_finished_seen": False,
        # Authoritative values from the explicit Setup/Step log lines, when
        # the binary was built with the run_solver / solve_lp_distributed_from_mps
        # patch. None on older binaries.
        "explicit_setup_time_s": None,
        "explicit_step_time_s": None,
        "explicit_time_per_step_ms": None,
        # Free-form details captured for failure classification. exit_code
        # is from run_sweep.sh's "### exit_code=" trailer. failure_mode is
        # one of the strings in _FAILURE_PATTERNS (see below) or None on
        # success. failure_excerpt is the offending log line, truncated.
        "exit_code": None,
        "failure_mode": None,
        "failure_excerpt": None,
        "parse_warnings": [],
    }

    # First sweep: classify failure by scanning for the FIRST matching
    # pattern in the log. We do this in addition to the field-by-field
    # parse below so a single read of the file is enough.
    with path.open("r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # exit_code is the LAST trailer line written by run_sweep.sh;
            # capture every time so we end with the final value.
            m = _HDR_FOOTER.match(line)
            if m:
                try:
                    rec["exit_code"] = int(m.group("ec"))
                except ValueError:
                    pass

            if rec["failure_mode"] is None:
                for label, pat in _FAILURE_PATTERNS:
                    if pat.search(line):
                        rec["failure_mode"] = label
                        # Trim to keep CSVs sane.
                        excerpt = line.strip()
                        if len(excerpt) > 240:
                            excerpt = excerpt[:237] + "..."
                        rec["failure_excerpt"] = excerpt
                        break

            m = _HDR.match(line)
            if m:
                k, v = m.group("key"), m.group("val")
                rec[k] = v
                continue

            m = _DIMS.search(line)
            if m:
                rec["n_cstr"] = int(m.group("n_cstr"))
                rec["n_vars"] = int(m.group("n_vars"))
                rec["n_int"] = int(m.group("n_int"))
                rec["nnz"] = int(m.group("nnz"))
                continue

            m = _ITER_ROW.match(line)
            if m:
                it = int(m.group("iter"))
                el = float(m.group("elapsed"))
                rec["n_iter_rows"] += 1
                if rec["first_iter"] is None:
                    rec["first_iter"] = it
                    rec["first_iter_elapsed_s"] = el
                rec["last_iter"] = it
                rec["last_iter_elapsed_s"] = el
                continue

            if _PDLP_FINISHED.match(line):
                rec["pdlp_finished_seen"] = True
                continue

            m = _TERMINAL.match(line)
            if m:
                rec["status"] = m.group("status")
                try:
                    rec["objective"] = float(m.group("objective"))
                except ValueError:
                    rec["objective"] = None
                rec["iterations"] = int(m.group("iterations"))
                rec["total_time_s"] = float(m.group("total_time"))
                continue

            m = _SETUP_TIME.match(line)
            if m:
                rec["explicit_setup_time_s"] = float(m.group("setup"))
                continue

            m = _STEP_TIME.match(line)
            if m:
                rec["explicit_step_time_s"] = float(m.group("step"))
                rec["explicit_time_per_step_ms"] = float(m.group("ms_per_step"))
                # The binary already reports setup here too; if we somehow
                # missed the dedicated "Setup time:" line (truncated log,
                # different filter), still pick it up.
                if rec["explicit_setup_time_s"] is None:
                    rec["explicit_setup_time_s"] = float(m.group("setup"))
                continue

            m = _AUTO_DETECT.search(line)
            if m:
                rec["auto_detected_gpus"] = int(m.group("n"))
                continue

            m = _METIS.search(line)
            if m:
                rec["metis_time_s"] = float(m.group("dt"))
                rec["metis_edge_cut"] = int(m.group("edge_cut"))
                rec["metis_nb_parts"] = int(m.group("nb_parts"))
                continue

    # --- typed conversions for header values --------------------------------
    if rec["n_gpus_requested"] is not None:
        try:
            rec["n_gpus_requested"] = int(rec["n_gpus_requested"])
        except (TypeError, ValueError):
            pass

    # --- "is_warmup" classification ----------------------------------------
    # run_sweep.sh writes rep=W0, W1, ... for warmups and 1, 2, ... for
    # timed reps. Honor that explicitly; do not infer from filename.
    rep = rec.get("rep")
    rec["is_warmup"] = isinstance(rep, str) and rep.startswith("W")

    # --- derived fields -----------------------------------------------------
    # setup_time_s := wall-clock elapsed when the main PDLP loop is about to
    # take its first step. Two sources, in priority order:
    #   1. The bullshit_mostovoi_cuopt patch emits an explicit
    #      "Setup time: %.3fs (before PDLP iteration loop)" line right at
    #      that boundary inside pdlp_solver_t::run_solver. Authoritative.
    #   2. Older binaries: use first_iter_elapsed_s, which is the elapsed
    #      reported on the first per-major-iter row. The print happens
    #      BEFORE a step (see pdlp.cu major-iter check), so for a normal
    #      run the first row fires at iter=0 and this approximates the
    #      true setup. May overcount by ~1 iter if the first row is later.
    if rec["explicit_setup_time_s"] is not None:
        rec["setup_time_s"] = rec["explicit_setup_time_s"]
        rec["setup_time_source"] = "explicit"
    else:
        rec["setup_time_s"] = rec["first_iter_elapsed_s"]
        rec["setup_time_source"] = "first_iter_row"

    # step_time_s := time spent inside the PDLP iteration loop (the part
    # that strong/weak scaling actually cares about). The patched binary
    # emits this directly via "Step time: %.3fs"; otherwise derive it as
    # total_time_s - setup_time_s.
    if rec["explicit_step_time_s"] is not None:
        rec["step_time_s"] = rec["explicit_step_time_s"]
    elif (
        rec["total_time_s"] is not None
        and rec["setup_time_s"] is not None
        and rec["total_time_s"] >= rec["setup_time_s"]
    ):
        rec["step_time_s"] = rec["total_time_s"] - rec["setup_time_s"]
    else:
        rec["step_time_s"] = None

    if rec["explicit_time_per_step_ms"] is not None:
        rec["time_per_step_ms"] = rec["explicit_time_per_step_ms"]
    elif rec["step_time_s"] is not None and rec["iterations"]:
        rec["time_per_step_ms"] = (rec["step_time_s"] / rec["iterations"]) * 1000.0
    else:
        rec["time_per_step_ms"] = None

    # Secondary: a between-major-iter delta. Useful when the first row
    # didn't print at iter=0 (e.g. warm-start) so step_time_s would
    # overcount; this fallback is a delta within the table itself.
    if (
        rec["first_iter_elapsed_s"] is not None
        and rec["last_iter_elapsed_s"] is not None
        and rec["last_iter"] is not None
        and rec["first_iter"] is not None
        and rec["last_iter"] > rec["first_iter"]
    ):
        loop_dt = rec["last_iter_elapsed_s"] - rec["first_iter_elapsed_s"]
        loop_iters = rec["last_iter"] - rec["first_iter"]
        rec["loop_time_s"] = loop_dt
        rec["loop_iters"] = loop_iters
        rec["time_per_iter_ms"] = (loop_dt / loop_iters) * 1000.0 if loop_iters > 0 else None
    else:
        rec["loop_time_s"] = None
        rec["loop_iters"] = None
        rec["time_per_iter_ms"] = None

    # --- final failure classification --------------------------------------
    # A run is a SUCCESS iff it produced a Status terminal line with a
    # status string that is not one of the failure modes above.
    # Otherwise pick the most informative reason in order of evidence:
    #   - explicit pattern match found while scanning
    #   - cuopt_cli exit_code != 0 → "Crashed"
    #   - no Status line at all → "Truncated"
    if rec["status"] is None and rec["failure_mode"] is None:
        if rec.get("exit_code") not in (None, 0):
            rec["failure_mode"] = "Crashed"
            rec["failure_excerpt"] = f"cuopt_cli exited with status {rec['exit_code']}"
        else:
            rec["failure_mode"] = "Truncated"
            rec["failure_excerpt"] = "no terminal Status line and no exit_code"

    rec["succeeded"] = (rec["failure_mode"] is None) and (rec["status"] is not None)

    # --- sanity warnings ----------------------------------------------------
    warns = list(rec["parse_warnings"])
    if rec["status"] is None and rec["failure_mode"] in (None, "Truncated"):
        warns.append("missing terminal Status line (truncated log?)")
    if rec["iterations"] is None and rec["succeeded"]:
        warns.append("missing iteration count on a successful run")
    if not rec["pdlp_finished_seen"] and rec["succeeded"]:
        warns.append("missing 'PDLP finished' sentinel on a successful run")
    if rec["n_iter_rows"] < 2:
        warns.append(
            f"only {rec['n_iter_rows']} iter-table row(s); cannot derive per-iter time"
        )
    if (
        rec.get("first_iter") not in (None, 0)
        and rec.get("setup_time_source") != "explicit"
    ):
        warns.append(
            f"first iter-table row was at iter={rec['first_iter']}, not 0; "
            "step_time_s overcounts (it folds in iter [0, first_iter)). "
            "Rebuild cuopt with the run_solver patch to get the explicit "
            "Setup time / Step time log lines."
        )
    rec["parse_warnings"] = warns

    return rec


# ---- CLI --------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="Parse a cuopt_cli log emitted by run_sweep.sh.")
    ap.add_argument("path", type=Path, help="Path to log file.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print the JSON output.")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"log not found: {args.path}", file=sys.stderr)
        return 1
    rec = parse_log(args.path)
    if args.pretty:
        print(json.dumps(rec, indent=2, sort_keys=True))
    else:
        print(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
