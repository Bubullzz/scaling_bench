#!/usr/bin/env python3
"""
plot_endtoend_presolved.py -- run the standard `plot_endtoend` "all" view
(cuopt-distributed + D-PDLP + cuopt-base side by side) but restricted to
the 5 instances we solve using their Gurobi-presolved variant.

These 5 instances share a subtle plotting quirk: for cuopt-* the setup
bar of the presolved run STILL includes the ~small PSLP presolve time
because presolve=0 only skips the reducer, not the model transforms;
for D-PDLP the ` --presolve 0` flag actually disables PSLP so the setup
bar is much thinner. Isolating them in their own figure makes that
apples-to-apples comparison legible without the huge dynamic range of
the full 25-instance chart drowning them.

Presolved instances (as of Jul 2026):
    - C5_bigger_sanitized_gurobi_presolved
    - ELMOD_876_10_noVEnames_gurobi_presolved
    - design_match_gurobi_presolved
    - psr_100_gurobi_presolved
    - qap-tho-150_gurobi_presolved

Usage:
    python plot_endtoend_presolved.py                 # tol=1e-4
    python plot_endtoend_presolved.py --tol 1e-6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plot_endtoend import (
    HERE, load_rows, plot_view,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tol", choices=("1e-4", "1e-6"), default="1e-4")
    p.add_argument("--csv", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    csv_path = Path(args.csv or (HERE / f"results_endtoend_tol{args.tol}.csv"))
    if not csv_path.is_file():
        sys.exit(f"CSV not found: {csv_path}. Run endtoend_build_csv.py first.")
    out_dir = Path(args.out_dir or (HERE / "plots"))

    rows = load_rows(csv_path)

    # Filter to only the Gurobi-presolved variants. The suffix
    # convention is set by presolve.py in gurobi_things/: it always
    # appends `_gurobi_presolved` to the stem before `.mps`. Using
    # substring match rather than a hardcoded whitelist means new
    # presolved instances (e.g. if we ever presolve qap-wil-100)
    # automatically show up without editing this script.
    rows = [r for r in rows if "gurobi_presolved" in r.instance]
    if not rows:
        sys.exit("no *_gurobi_presolved rows in CSV; nothing to plot.")

    out_path = out_dir / f"endtoend_tol{args.tol}_presolved_only.png"
    plot_view(rows, "all", out_path, tol=args.tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
