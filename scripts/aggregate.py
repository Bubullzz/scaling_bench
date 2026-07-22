#!/usr/bin/env python3
"""
scaling_bench/scripts/aggregate.py

Walk runs/, parse every *.log with parse_log.parse_log, and produce two CSVs:

  results.csv          - one row per timed rep (warm-ups still included with
                         is_warmup=True so the CSV is self-describing, but
                         get filtered out of the summary).
  results_summary.csv  - one row per (problem, n_gpus) with median / mean /
                         stdev / CoV of the headline scaling metrics over
                         timed reps only.

Stdlib only.
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

# Local import without packaging.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from parse_log import parse_log  # noqa: E402


# Columns we write in results.csv. Order matters (CSV header order).
# Note: step_time_s / time_per_step_ms are the *headline* metrics for scaling
# (they exclude setup, which grows with N and isn't representative of
# per-iteration cost). loop_time_s / time_per_iter_ms are the older
# between-major-iter delta, kept as a sanity-check secondary metric.
ROW_COLUMNS = [
    "problem",
    "stem",
    "n_gpus_requested",
    "rep",
    "is_warmup",
    "status",
    "iterations",
    "n_cstr",
    "n_vars",
    "nnz",
    "setup_time_s",
    "setup_time_source",   # "explicit" (patched binary) or "first_iter_row" (fallback)
    "step_time_s",
    "time_per_step_ms",
    "total_time_s",
    "first_iter",
    "last_iter",
    "n_iter_rows",
    "loop_time_s",
    "loop_iters",
    "time_per_iter_ms",
    "metis_time_s",
    "metis_edge_cut",
    "objective",
    # Failure classification (None / "" on success).
    "succeeded",
    "failure_mode",
    "failure_excerpt",
    "exit_code",
    "partition_file",
    "cuda_visible_devices",
    "auto_detected_gpus",
    "pdlp_finished_seen",
    "started",
    "finished",
    "log_path",
    "parse_warnings",
]

SUMMARY_COLUMNS = [
    "problem",
    "stem",
    "n_gpus",
    "n_reps_timed",
    "iters_consistent",      # True if all reps agree on iteration count
    "iters_min",
    "iters_max",
    "status_consensus",      # status string if all reps agree, else "MIXED:<csv>"
    # PRIMARY scaling metric: time spent inside the PDLP step loop.
    "step_time_s_median",
    "step_time_s_mean",
    "step_time_s_stdev",
    "step_time_s_cov",
    "time_per_step_ms_median",
    "time_per_step_ms_mean",
    "time_per_step_ms_stdev",
    "time_per_step_ms_cov",
    # Setup cost (partition + NCCL bootstrap + scaling + sigma_max).
    "setup_time_s_median",
    "setup_fraction_median",  # setup_time_s / total_time_s, the Amdahl-style serial share
    # Secondary / sanity-check: between-major-iter table delta.
    "loop_time_s_median",
    "time_per_iter_ms_median",
    "time_per_iter_ms_cov",
    # Full wall-clock (setup + steps). Kept for completeness.
    "total_time_s_median",
    "n_cstr",
    "n_vars",
    "nnz",
    # Failure rollup: counts of OK vs failed reps per (problem, N), and a
    # comma-separated list of distinct failure_mode values that showed up.
    "n_reps_ok",
    "n_reps_failed",
    "failure_modes",
    "parse_warnings_any",
]


def _stats(xs):
    """median, mean, stdev, cov over non-None values. (None, None, None, None) if empty."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None, None
    med = st.median(xs)
    mean = st.fmean(xs)
    sd = st.stdev(xs) if len(xs) >= 2 else 0.0
    cov = (sd / mean) if mean else None
    return med, mean, sd, cov


def _walk_logs(runs_dir: Path):
    for p in sorted(runs_dir.rglob("*.log")):
        yield p


def main() -> int:
    runs_dir = HERE.parent / "runs"
    out_dir = HERE.parent
    rows_path = out_dir / "results.csv"
    summary_path = out_dir / "results_summary.csv"

    if not runs_dir.exists():
        print(f"no runs directory at {runs_dir}", file=sys.stderr)
        return 1

    records = []
    for p in _walk_logs(runs_dir):
        try:
            rec = parse_log(p)
        except Exception as e:
            print(f"parse failed for {p}: {e}", file=sys.stderr)
            continue
        records.append(rec)

    if not records:
        print("no logs parsed", file=sys.stderr)
        return 1

    # ---- write per-run rows ------------------------------------------------
    with rows_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ROW_COLUMNS)
        for r in records:
            row = []
            for col in ROW_COLUMNS:
                v = r.get(col)
                if isinstance(v, list):
                    v = ";".join(map(str, v))
                row.append(v)
            w.writerow(row)

    # ---- group + summarize over timed reps --------------------------------
    by_pair = defaultdict(list)  # (problem, n_gpus) -> [recs]
    for r in records:
        if r.get("is_warmup"):
            continue
        if r.get("n_gpus_requested") is None or r.get("problem") is None:
            continue
        by_pair[(r["problem"], r["n_gpus_requested"])].append(r)

    n_timed = sum(1 for r in records if not r.get("is_warmup"))
    n_failed = sum(1 for r in records if not r.get("is_warmup") and not r.get("succeeded"))

    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLUMNS)
        for (problem, n_gpus), reps in sorted(by_pair.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            ok_reps = [r for r in reps if r.get("succeeded")]
            fail_reps = [r for r in reps if not r.get("succeeded")]
            # Only compute timing statistics over reps that actually
            # succeeded - failed reps have None timings or nonsense.
            iters = [r.get("iterations") for r in ok_reps if r.get("iterations") is not None]
            statuses = sorted({str(r.get("status")) for r in ok_reps if r.get("status") is not None})
            step_times = [r.get("step_time_s") for r in ok_reps]
            tps = [r.get("time_per_step_ms") for r in ok_reps]
            loop_times = [r.get("loop_time_s") for r in ok_reps]
            tpi = [r.get("time_per_iter_ms") for r in ok_reps]
            setup_times = [r.get("setup_time_s") for r in ok_reps]
            total_times = [r.get("total_time_s") for r in ok_reps]
            setup_fracs = [
                (r.get("setup_time_s") / r.get("total_time_s"))
                for r in ok_reps
                if r.get("setup_time_s") is not None
                and r.get("total_time_s") not in (None, 0)
            ]
            fail_modes = sorted({
                str(r.get("failure_mode"))
                for r in fail_reps
                if r.get("failure_mode") is not None
            })

            step_med, step_mean, step_sd, step_cov = _stats(step_times)
            tps_med, tps_mean, tps_sd, tps_cov = _stats(tps)
            loop_med, _, _, _ = _stats(loop_times)
            tpi_med, _, _, tpi_cov = _stats(tpi)
            setup_med, _, _, _ = _stats(setup_times)
            setup_frac_med, _, _, _ = _stats(setup_fracs)
            total_med, _, _, _ = _stats(total_times)

            iters_consistent = bool(iters) and len(set(iters)) == 1
            iters_min = min(iters) if iters else None
            iters_max = max(iters) if iters else None
            status_consensus = (
                statuses[0] if len(statuses) == 1
                else f"MIXED:{','.join(statuses)}" if statuses
                else None
            )
            any_warns = any(r.get("parse_warnings") for r in reps)
            stem = next((r.get("stem") for r in reps if r.get("stem")), None)
            n_cstr = next((r.get("n_cstr") for r in reps if r.get("n_cstr") is not None), None)
            n_vars = next((r.get("n_vars") for r in reps if r.get("n_vars") is not None), None)
            nnz = next((r.get("nnz") for r in reps if r.get("nnz") is not None), None)

            w.writerow([
                problem,
                stem,
                n_gpus,
                len(ok_reps),
                iters_consistent,
                iters_min,
                iters_max,
                status_consensus,
                step_med, step_mean, step_sd, step_cov,
                tps_med, tps_mean, tps_sd, tps_cov,
                setup_med,
                setup_frac_med,
                loop_med,
                tpi_med,
                tpi_cov,
                total_med,
                n_cstr, n_vars, nnz,
                len(ok_reps),
                len(fail_reps),
                ",".join(fail_modes),
                any_warns,
            ])

    print(
        f"Parsed {len(records)} log file(s); {n_timed} timed "
        f"({n_timed - n_failed} ok, {n_failed} failed); wrote:\n"
        f"  {rows_path}\n  {summary_path}"
    )
    if n_failed:
        # Print one line per (problem, N) that had any failure, with
        # mode + a short excerpt. Saves the user from grepping logs.
        print("\nFailed runs:")
        for (problem, n_gpus), reps in sorted(by_pair.items()):
            fail_reps = [r for r in reps if not r.get("succeeded")]
            if not fail_reps:
                continue
            stem = next((r.get("stem") for r in fail_reps if r.get("stem")), None)
            for r in fail_reps:
                mode = r.get("failure_mode") or "?"
                excerpt = r.get("failure_excerpt") or ""
                print(f"  [{mode}] {stem} N={n_gpus} rep={r.get('rep')}: {excerpt}")
                print(f"           log: {r.get('log_path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
