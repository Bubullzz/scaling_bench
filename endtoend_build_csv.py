#!/usr/bin/env python3
"""
endtoend_build_csv.py -- consolidate per-run logs under runs_endtoend/<tol>/
into a flat results_endtoend_tol<tol>.csv, one row per (solver, instance,
n_gpus).

Independent of endtoend_bench.py: it just scans the on-disk logs and writes
the CSV. Safe to run while workers are still producing new runs -- rerun
whenever you want a fresh CSV (or before regenerating plots).

Rows with no `### exit_code=...` footer are counted as `partial` (worker
still running or interrupted) and skipped from the CSV so the plot doesn't
show half-finished data.

Usage:
    python endtoend_build_csv.py                      # tol=1e-4 by default
    python endtoend_build_csv.py --tol 1e-6
    python endtoend_build_csv.py --tol 1e-6 --solver dpdlp
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import bench
from bench import _cuopt_parse, _dpdlp_parse, _stem_of


HERE = Path(__file__).resolve().parent
EE_RUNS_DIR = HERE / "runs_endtoend"


# Columns to emit, in a fixed order so downstream (plot script, spreadsheets)
# don't have to guess. Missing values become empty strings.
CSV_COLS = [
    "solver", "instance", "n_gpus", "tol",
    "status", "exit_code",
    "total_s", "presolve_s", "setup_s", "step_s",
    "iterations",
    "gpu_peak_mb", "gpu_peak2_mb",
]


RX_EXIT   = re.compile(r"^### exit_code=(?P<ec>[0-9-]+)\s*$", re.M)
RX_START  = re.compile(r"^### started=(?P<t>\S+)\s*$",       re.M)
RX_END    = re.compile(r"^### finished=(?P<t>\S+)\s*$",      re.M)
RX_GPU_PK = re.compile(r"^### gpu_peak_mb=(?P<v>\S+)\s*$",   re.M)
RX_GPU_P2 = re.compile(r"^### gpu_peak2_mb=(?P<v>\S+)\s*$",  re.M)
RX_ARGV   = re.compile(r"^### argv=(?P<argv>.+)$",           re.M)
RX_INST   = re.compile(r"^### instance=(?P<i>.+)$",          re.M)
RX_SOLVER = re.compile(r"^### solver=(?P<s>.+)$",            re.M)
RX_NGPUS  = re.compile(r"^### n_gpus=(?P<n>\d+)",            re.M)


def _n_gpus_from_stem_suffix(name: str) -> int | None:
    """Extract N from '<stem>__N<gpus>[.log]' if it matches; None otherwise."""
    m = re.search(r"__N(\d+)(?:\.log)?$", name)
    return int(m.group(1)) if m else None


def _find_run_logs(root: Path, solver_filter: str | None) -> list[tuple[str, Path]]:
    """Return (solver_name, log_path) pairs under one tolerance root.
    solver_name is derived from the parent directory name, so /cuopt-base/
    -> solver='cuopt-base', /dpdlp/ -> solver='dpdlp', etc."""
    out: list[tuple[str, Path]] = []
    if not root.is_dir():
        return out
    for solver_dir in sorted(root.iterdir()):
        if not solver_dir.is_dir():
            continue
        solver = solver_dir.name
        if solver_filter and solver != solver_filter:
            continue
        if solver == "dpdlp":
            # Per-run subdir with run.log inside.
            for rd in sorted(solver_dir.iterdir()):
                if rd.is_dir():
                    lp = rd / "run.log"
                    if lp.is_file():
                        out.append((solver, lp))
        else:
            # Flat *.log files for cuopt / cuopt-basic.
            for lp in sorted(solver_dir.glob("*.log")):
                out.append((solver, lp))
    return out


def _parse_log(solver: str, log_path: Path) -> dict:
    """Parse one solver's log file. Dispatches to bench's parser; adds
    minimal fallback for header-only crashed runs."""
    text = log_path.read_text(errors="replace")
    metrics: dict = {}

    if solver == "dpdlp":
        # bench's dpdlp parser wants (log_path, run_dir) so it can pick up
        # summary.txt if present.
        metrics = _dpdlp_parse(log_path, log_path.parent)
    else:
        # cuopt / cuopt-basic: bench parser signature (log_path, run_dir).
        metrics = _cuopt_parse(log_path, log_path.parent)

    # Fill in bookkeeping fields not covered by the solver parsers.
    if not metrics.get("exit_code"):
        m = RX_EXIT.search(text)
        if m:
            metrics["exit_code"] = m.group("ec")
    if not metrics.get("gpu_peak_mb"):
        m = RX_GPU_PK.search(text)
        if m:
            metrics["gpu_peak_mb"] = m.group("v")
    if not metrics.get("gpu_peak2_mb"):
        m = RX_GPU_P2.search(text)
        if m:
            metrics["gpu_peak2_mb"] = m.group("v")

    # Instance path: try to recover from the log header first, fall back
    # to the parent stem so we still get *something* in the CSV.
    m = RX_INST.search(text)
    if m:
        metrics["instance"] = Path(m.group("i")).name
    m = RX_NGPUS.search(text)
    if m:
        metrics["n_gpus"] = int(m.group("n"))
    return metrics


def _stem_from_path(solver: str, log_path: Path) -> tuple[str, int | None]:
    """Fallback: derive (stem, n_gpus) from the filename/dirname layout."""
    if solver == "dpdlp":
        # dpdlp/<stem>__N<n>/run.log
        dirname = log_path.parent.name
    else:
        # cuopt-*/<stem>__N<n>.log
        dirname = log_path.name.rsplit(".log", 1)[0]
    n = _n_gpus_from_stem_suffix(dirname)
    stem = re.sub(r"__N\d+$", "", dirname)
    return stem, n


def _has_footer(log_path: Path) -> bool:
    """True iff the log contains an `### exit_code=` footer -- the marker
    that the wrapper wrote the run's metadata tail before exiting."""
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            # Read last 4 KiB; the footer is always in the last 500 bytes.
            f.seek(max(0, size - 4096))
            tail = f.read().decode(errors="replace")
        return "### exit_code=" in tail
    except OSError:
        return False


def build_csv(tol: str, solver_filter: str | None, out_path: Path) -> None:
    root = EE_RUNS_DIR / tol
    logs = _find_run_logs(root, solver_filter)
    print(f"endtoend_build_csv (tol={tol}): scanned {len(logs)} log(s) "
          f"under {root}")

    rows: list[dict] = []
    partial = 0
    skipped = 0
    seen_keys: set[tuple[str, str, int]] = set()
    dupes = 0

    for solver, lp in logs:
        if not _has_footer(lp):
            partial += 1
            continue
        try:
            m = _parse_log(solver, lp)
        except Exception as e:
            print(f"  !! failed to parse {lp}: {e}", file=sys.stderr)
            skipped += 1
            continue

        if not m.get("instance") or not m.get("n_gpus"):
            stem, n = _stem_from_path(solver, lp)
            m.setdefault("instance", f"{stem}.mps")
            m.setdefault("n_gpus", n or 0)

        key = (solver, str(m.get("instance") or ""), int(m.get("n_gpus") or 0))
        if key in seen_keys:
            dupes += 1
            continue
        seen_keys.add(key)

        row = {c: m.get(c, "") for c in CSV_COLS}
        row["solver"] = solver
        row["tol"] = tol
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"  wrote   : {len(rows)} row(s) -> {out_path}")
    print(f"  partial : {partial} (no '### exit_code=' footer, run not finished)")
    print(f"  skipped : {skipped} for filtered solver(s), "
          f"{sum(1 for _ in logs) - len(rows) - partial - dupes} "
          f"unparseable / missing metadata")
    print(f"  dupes   : {dupes} duplicate (solver,instance,n_gpus) keys")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tol", choices=("1e-4", "1e-6"), default="1e-4",
                   help="which per-tolerance subdir to consolidate")
    p.add_argument("--solver", default=None,
                   choices=("cuopt-distributed", "cuopt-base", "dpdlp"),
                   help="only include this solver's logs")
    p.add_argument("--out", default=None,
                   help="output CSV path (default: results_endtoend_tol<tol>.csv)")
    args = p.parse_args()

    out = Path(args.out) if args.out else HERE / f"results_endtoend_tol{args.tol}.csv"
    build_csv(args.tol, args.solver, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
