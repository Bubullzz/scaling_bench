#!/usr/bin/env python3
"""
build_csv.py - merge every run log under runs/ into a single results.csv.

This is the *consumer* in the bench.py/build_csv.py producer/consumer
split. bench.py workers only write per-run logs; this script reads them
and emits a CSV. Running it has no effect on what work bench.py will
schedule (all of that is driven off the log files themselves).

Run it any time you want a fresh consolidated view:

    python build_csv.py                  # write ./results.csv from ./runs/
    python build_csv.py -o foo.csv       # custom output path
    python build_csv.py --runs-dir /shared/runs  # custom runs/ dir

It is safe to:
  - run while bench.py workers are still going (you'll get a snapshot;
    re-run for the latest).
  - run from a different host than the workers, as long as that host
    can read the shared runs/ directory.
  - run multiple times in a row -- the CSV is rewritten from scratch
    each time via tmp-file + atomic rename.

Logs that have no '### exit_code=' footer (interrupted runs) are
silently skipped. Logs whose '### solver=' or '### n_gpus=' header is
missing or unparseable are reported in the final summary but don't
abort the run.

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
from pathlib import Path

# Reuse bench.py's parsers, solver registry, CSV schema, and row builder.
# That gives us a single source of truth for log format and CSV columns.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from bench import (  # noqa: E402
    CSV_COLUMNS,
    RUNS_DIR as DEFAULT_RUNS_DIR,
    SOLVERS,
    _csv_row_for,
    _parse_log_metadata,
    _stem_of,
)


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "-o", "--output", type=Path, default=Path("results.csv"),
        help="path to the output CSV. Default: ./results.csv",
    )
    p.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
        help=f"path to the runs/ directory to scan. "
             f"Default: {DEFAULT_RUNS_DIR}",
    )
    p.add_argument(
        "--solver", action="append", default=None,
        choices=sorted(SOLVERS.keys()),
        help="restrict output to one or more solvers. Repeatable. "
             "Default: include every solver registered in bench.SOLVERS.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="print one line per accepted log",
    )
    return p.parse_args()


def main() -> int:
    args = parse_cli()

    runs_dir: Path = args.runs_dir
    if not runs_dir.is_dir():
        sys.exit(f"runs dir not found: {runs_dir}")

    active_solvers = set(args.solver) if args.solver else set(SOLVERS.keys())

    rows: list[list] = []
    n_inspected = 0
    n_added = 0
    n_partial = 0          # log exists but no '### exit_code=' footer
    n_skipped_solver = 0   # solver not in --solver filter (or not registered)
    n_skipped_meta = 0     # log has no parseable '### key=value' metadata
    n_dup = 0              # same (solver, instance, n_gpus) seen twice
    seen: set[tuple[str, str, int]] = set()

    # We sort logs deterministically so the CSV ordering is reproducible
    # even when re-running build_csv.py against the same runs/ dir.
    for log_path in sorted(runs_dir.rglob("*.log")):
        n_inspected += 1
        meta = _parse_log_metadata(log_path)
        if meta is None:
            n_skipped_meta += 1
            continue

        solver_name = meta.get("solver")
        if solver_name is None or solver_name not in SOLVERS:
            n_skipped_meta += 1
            continue
        if solver_name not in active_solvers:
            n_skipped_solver += 1
            continue

        try:
            n_gpus = int(meta["n_gpus"])
        except (KeyError, ValueError):
            n_skipped_meta += 1
            continue

        instance_path = meta.get("instance")
        if not instance_path:
            n_skipped_meta += 1
            continue
        instance_name = Path(instance_path).name

        if "exit_code" not in meta:
            n_partial += 1
            continue

        key = (solver_name, instance_name, n_gpus)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)

        solver = SOLVERS[solver_name]
        # parse_log needs run_dir to find dpdlp's _summary.txt; cuopt
        # ignores it. We recompute it from (instance, n_gpus) instead of
        # trusting the log's location so we don't get tripped up by logs
        # in legacy locations.
        rd = solver.run_dir(instance_path, n_gpus)
        parsed = solver.parse_log(log_path, rd)
        parsed["gpu_peak_mb"]  = meta.get("gpu_peak_mb")
        parsed["gpu_peak2_mb"] = meta.get("gpu_peak2_mb")

        row = _csv_row_for(solver_name, instance_name, n_gpus,
                           parsed, meta.get("exit_code", ""), log_path)
        rows.append(row)
        n_added += 1

        if args.verbose:
            fmt = lambda x: "" if x is None else x
            print(f"  + {solver_name:17s} {_stem_of(instance_name):28s} N={n_gpus}  "
                  f"step={fmt(parsed['step_s'])}s  "
                  f"gpu_peak={fmt(parsed['gpu_peak_mb'])} "
                  f"peak2={fmt(parsed['gpu_peak2_mb'])}")

    # Sort the output by (solver, instance, n_gpus) for a stable layout.
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    # Write atomically: tmp file in the same directory + os.replace.
    # Same-directory rename is the only thing POSIX guarantees to be
    # atomic, and it works just as well on NFS as on local FS.
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(
        f"{out_path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
    )
    with tmp_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        w.writerows(rows)
    os.replace(tmp_path, out_path)

    print(
        f"\nbuild_csv: scanned {n_inspected} log(s) under {runs_dir}\n"
        f"  wrote   : {n_added} row(s) -> {out_path}\n"
        f"  partial : {n_partial} (no '### exit_code=' footer, run not finished)\n"
        f"  skipped : {n_skipped_solver} for filtered solver(s), "
        f"{n_skipped_meta} unparseable / missing metadata\n"
        f"  dupes   : {n_dup} duplicate (solver,instance,n_gpus) keys "
        f"(first one won; check for stale logs in legacy locations)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
