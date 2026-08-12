#!/usr/bin/env python3
"""plot_run_status.py -- overview of end-to-end sweep progress.

One figure covering BOTH tolerances (1e-4 and 1e-6):
  - summary stacked bars (counts per solver × status)
  - heatmaps instance × solver coloured by outcome

Statuses:
  Optimal    -- converged
  Time Limit -- hit the 3600s wall (or status TIME_LIMIT)
  Crash      -- finished with empty/non-optimal status or non-zero exit
  Hang skip  -- manually marked hang (exit_code=124), so the sweep skips it
  Partial    -- log on disk but no '### exit_code=' footer (interrupted)
  Missing    -- not started yet (still to compute)

Usage:
    python plot_run_status.py
    python plot_run_status.py --out plots/run_status.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

import bench
import endtoend_bench as ee


HERE = Path(__file__).resolve().parent

SOLVERS = ["cuopt-distributed", "dpdlp", "cuopt-base"]
SOLVER_SHORT = {
    "cuopt-distributed": "cuOpt-dist\n(N=8)",
    "dpdlp":             "D-PDLP\n(N=8)",
    "cuopt-base":        "cuOpt-base\n(N=1)",
}
TOLS = ["1e-4", "1e-6"]

# Discrete status codes for the heatmap
ST_OPTIMAL = 0
ST_TL      = 1
ST_CRASH   = 2
ST_HANG    = 3
ST_PARTIAL = 4
ST_MISSING = 5

ST_LABEL = {
    ST_OPTIMAL: "Optimal",
    ST_TL:      "Time Limit",
    ST_CRASH:   "Crash",
    ST_HANG:    "Hang (skipped)",
    ST_PARTIAL: "Partial (interrupted)",
    ST_MISSING: "To compute",
}
ST_COLOR = {
    ST_OPTIMAL: "#76B900",  # NVIDIA green
    ST_TL:      "#F0A202",  # amber
    ST_CRASH:   "#C62828",  # red
    ST_HANG:    "#6A1B9A",  # purple
    ST_PARTIAL: "#546E7A",  # blue-grey
    ST_MISSING: "#D0D5DB",  # light grey
}

# Short display labels (extend plot_speedup_1v1.DISPLAY)
DISPLAY = {
    "C5_bigger_sanitized": "C5-bigger",
    "C5_bigger_sanitized_gurobi_presolved": "C5-bigger*",
    "C5_baseline_sanitized": "C5-baseline",
    "Dual2_5000": "Dual2-5000",
    "ELMOD_876_10_noVEnames": "ELMOD-876",
    "ELMOD_876_10_noVEnames_gurobi_presolved": "ELMOD-876*",
    "amazon_lp003": "amazon-lp003",
    "amazon_lp004": "amazon-lp004",
    "dlr2": "dlr2",
    "heat-source-easy": "heat-source-easy",
    "heat-source-hard": "heat-source-hard",
    "mediterranean-shipping": "med-shipping",
    "multicommodity-flow-instance_2500_100_500": "mcf-2500×100×500",
    "multicommodity-flow-instance_5000_100_250": "mcf-5000×100×250",
    "multicommodity-flow-instance_5000_50_500": "mcf-5000×50×500",
    "production-imventory": "production-inventory",
    "psr_100": "psr-100",
    "psr_100_gurobi_presolved": "psr-100*",
    "qap-tho-150": "qap-tho-150",
    "qap-tho-150_gurobi_presolved": "qap-tho-150*",
    "qap-wil-100": "qap-wil-100",
    "supply-chain": "supply-chain",
    "tsp-gaia-10m": "tsp-gaia-10m",
    "design_match": "design-match",
    "design_match_gurobi_presolved": "design-match*",
    "zib03": "zib03",
    "world-shipping": "world-shipping",
    "BEAM_4032_11_8_CLI": "BEAM-4032",
    "zen-garden-eur-PI-28-200ts": "zen-garden-200ts",
    "zen-garden-eur-PI-constrained-expansion-28-100ts": "zen-garden-exp-100ts",
    "zen-garden-eur-PI-no-storage-28-100ts": "zen-garden-nostore",
    "model_de_20_scenarios": "PSR6-20",
    "model_de_50_scenarios": "PSR6-50",
    "model_de_100_scenarios": "PSR6-100",
    "IESA-Opt-NL-10-1h": "IESA-10-1h",
    "IESA-Opt-NL-5-3h": "IESA-5-3h",
    "TIMES-STEM-15-1h": "TIMES-STEM-15-1h",
    "pypsa-de-elec-60-1h": "pypsa-de-60-1h",
    "pypsa-eur-elec-100-3h": "pypsa-eur-100-3h",
    "pypsa-eur-sec-50-24h": "pypsa-eur-sec-50",
    "ethos_fine_europe_60tp-175-720ts": "ethos-fine-eu",
    "times-ireland-noco2-40-1ts": "times-ireland",
}

TIME_LIMIT_S = 3600.0
TIME_LIMIT_TOL = 0.90


def _stem(instance: str) -> str:
    """Same strip as the rest of the bench (.mps[.gz|.bz2|.lz4])."""
    return bench._stem_of(instance.strip())


def _display(stem: str) -> str:
    return DISPLAY.get(stem, stem)


def _classify(status: str, exit_code: str, total_s: float | None,
              iterations: int | None) -> int:
    """Map a CSV row to a discrete status code."""
    # Manual hang marker (GNU timeout convention / our skip footer).
    if exit_code.strip() == "124":
        return ST_HANG

    s = (status or "").strip().upper()
    if s in ("OPTIMAL", "SOLVED"):
        # Non-zero exit after Optimal is still a crash (rare).
        if exit_code not in ("", "0"):
            return ST_CRASH
        return ST_OPTIMAL
    if s in ("TIME_LIMIT", "TIME LIMIT", "TIMELIMIT"):
        return ST_TL
    if (total_s is not None
            and total_s >= TIME_LIMIT_TOL * TIME_LIMIT_S
            and (iterations or 0) > 0):
        return ST_TL
    # Anything present in the CSV that isn't Optimal/TL is a finished failure.
    return ST_CRASH


def _as_float(x) -> float | None:
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _as_int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def expected_stems() -> list[str]:
    """Canonical instance stems in sweep order (NO_PRESOLVE_THESE = raw paths)."""
    return [bench._stem_of(p) for p in ee._all_instances()]


def _scan_partials(tol: str) -> set[tuple[str, str]]:
    """(solver, stem) pairs whose log exists but has no exit_code footer."""
    root = HERE / "runs_endtoend" / tol
    found: set[tuple[str, str]] = set()
    if not root.is_dir():
        return found
    for solver in SOLVERS:
        sdir = root / solver
        if not sdir.is_dir():
            continue
        for p in sdir.iterdir():
            if p.is_dir():
                # dpdlp layout: <stem>__N<n>/run.log
                log = p / "run.log"
                stem_n = p.name  # e.g. foo__N8
            elif p.suffix == ".log":
                log = p
                stem_n = p.stem  # e.g. foo__N8
            else:
                continue
            if "__N" not in stem_n:
                continue
            stem = stem_n.rsplit("__N", 1)[0]
            if not log.is_file():
                continue
            try:
                text = log.read_text(errors="replace")
            except OSError:
                continue
            if "### exit_code=" not in text:
                found.add((solver, stem))
    return found


def load_status_grid(csv_path: Path, stems: list[str], tol: str
                     ) -> dict[tuple[str, str], int]:
    """Return {(solver, stem): status_code}. Missing cells stay absent;
    partial on-disk logs (no footer) are marked ST_PARTIAL."""
    out: dict[tuple[str, str], int] = {}
    if csv_path.is_file():
        with csv_path.open(newline="") as f:
            for r in csv.DictReader(f):
                solver = r.get("solver", "").strip()
                stem = _stem(r.get("instance", ""))
                if solver not in SOLVERS:
                    continue
                code = _classify(
                    r.get("status", ""),
                    str(r.get("exit_code", "")),
                    _as_float(r.get("total_s")),
                    _as_int(r.get("iterations")),
                )
                out[(solver, stem)] = code
    # Partials only fill cells that aren't already finished in the CSV.
    for key in _scan_partials(tol):
        if key not in out:
            out[key] = ST_PARTIAL
    return out


def plot_status(grids: dict[str, dict[tuple[str, str], int]],
                stems: list[str],
                out_path: Path) -> None:
    n_inst = len(stems)
    n_sol = len(SOLVERS)

    fig = plt.figure(figsize=(14.5, max(10.0, 0.32 * n_inst + 3.5)))
    # Top: summary bars. Bottom: two heatmaps side by side.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 5.5],
                          hspace=0.28, wspace=0.18,
                          left=0.22, right=0.98, top=0.90, bottom=0.06)

    # ---- summary stacked bars ----
    for col, tol in enumerate(TOLS):
        ax = fig.add_subplot(gs[0, col])
        grid = grids[tol]
        counts = {s: Counter() for s in SOLVERS}
        for solver in SOLVERS:
            for stem in stems:
                code = grid.get((solver, stem), ST_MISSING)
                counts[solver][code] += 1

        x = range(n_sol)
        bottoms = [0] * n_sol
        order = [ST_OPTIMAL, ST_TL, ST_CRASH, ST_HANG, ST_PARTIAL, ST_MISSING]
        for code in order:
            vals = [counts[s][code] for s in SOLVERS]
            if sum(vals) == 0:
                continue
            ax.bar(x, vals, bottom=bottoms, color=ST_COLOR[code],
                   edgecolor="white", linewidth=0.4, width=0.65,
                   label=ST_LABEL[code] if col == 0 else None)
            for i, v in enumerate(vals):
                if v > 0:
                    light = code in (ST_MISSING, ST_PARTIAL, ST_TL)
                    ax.text(i, bottoms[i] + v / 2, str(v),
                            ha="center", va="center", fontsize=8,
                            color="#222" if light else "white",
                            fontweight="bold")
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        pending = (sum(counts[s][ST_MISSING] for s in SOLVERS)
                   + sum(counts[s][ST_PARTIAL] for s in SOLVERS))
        done = n_sol * n_inst - pending
        total = n_sol * n_inst
        ax.set_ylim(0, n_inst * 1.08)
        ax.set_xticks(list(x))
        ax.set_xticklabels([SOLVER_SHORT[s].split("\n")[0] for s in SOLVERS],
                           fontsize=9)
        ax.set_ylabel("# instances" if col == 0 else "")
        ax.set_title(f"tol={tol}   {done}/{total} done",
                     fontsize=11, fontweight="bold", pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False,
                      ncol=1)

    # ---- heatmaps ----
    cmap = ListedColormap([ST_COLOR[i] for i in range(6)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    for col, tol in enumerate(TOLS):
        ax = fig.add_subplot(gs[1, col])
        grid = grids[tol]
        mat = [[grid.get((solver, stem), ST_MISSING) for solver in SOLVERS]
               for stem in stems]

        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm,
                  interpolation="nearest")
        ax.set_xticks(range(n_sol))
        ax.set_xticklabels([SOLVER_SHORT[s] for s in SOLVERS], fontsize=9)
        ax.set_yticks(range(n_inst))
        if col == 0:
            ax.set_yticklabels([_display(s) for s in stems], fontsize=7.5)
        else:
            ax.set_yticklabels([])

        # Light grid lines between cells.
        ax.set_xticks([x - 0.5 for x in range(n_sol + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(n_inst + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Letter markers so colour-blind / B&W still readable.
        mark = {
            ST_OPTIMAL: "O",
            ST_TL:      "T",
            ST_CRASH:   "X",
            ST_HANG:    "H",
            ST_PARTIAL: "P",
            ST_MISSING: "",
        }
        for i, stem in enumerate(stems):
            for j, solver in enumerate(SOLVERS):
                code = mat[i][j]
                m = mark[code]
                if not m:
                    continue
                color = "#222" if code in (ST_TL, ST_PARTIAL) else "white"
                ax.text(j, i, m, ha="center", va="center",
                        fontsize=7, fontweight="bold", color=color)

        ax.set_title(f"tol={tol}", fontsize=11, fontweight="bold", pad=6)

    # Shared legend under the heatmaps.
    handles = [
        mpatches.Patch(color=ST_COLOR[c],
                       label=f"{ST_LABEL[c]}  ({mark[c] or '·'})")
        for c in (ST_OPTIMAL, ST_TL, ST_CRASH, ST_HANG, ST_PARTIAL, ST_MISSING)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6,
               fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.6, 0.985))

    n_inst = len(stems)
    n_runs = n_inst * len(SOLVERS)
    fig.suptitle(
        f"End-to-end sweep status  ({n_inst} instances × {len(SOLVERS)} solvers "
        f"= {n_runs} runs / tol)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def print_summary(grids: dict[str, dict[tuple[str, str], int]],
                  stems: list[str]) -> None:
    for tol in TOLS:
        grid = grids[tol]
        c = Counter(grid.get((s, st), ST_MISSING)
                    for s in SOLVERS for st in stems)
        total = len(SOLVERS) * len(stems)
        print(f"\n=== tol={tol}  ({total} expected) ===")
        for code in (ST_OPTIMAL, ST_TL, ST_CRASH, ST_HANG, ST_PARTIAL, ST_MISSING):
            print(f"  {ST_LABEL[code]:24s} {c[code]:3d}")
        pending = [(s, st, grid.get((s, st), ST_MISSING))
                   for s in SOLVERS for st in stems
                   if grid.get((s, st), ST_MISSING) in (ST_MISSING, ST_PARTIAL)]
        if pending:
            print(f"  still to compute ({len(pending)}):")
            by_solver: dict[str, list[str]] = {s: [] for s in SOLVERS}
            for s, st, code in pending:
                tag = " [partial]" if code == ST_PARTIAL else ""
                by_solver[s].append(_display(st) + tag)
            for s in SOLVERS:
                if by_solver[s]:
                    print(f"    {s}: {', '.join(by_solver[s])}")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=HERE / "plots" / "run_status.png",
                   help="output PNG (default: plots/run_status.png)")
    p.add_argument("--csv-dir", type=Path, default=HERE,
                   help="dir containing results_endtoend_tol*.csv")
    return p.parse_args()


def main() -> int:
    args = parse_cli()
    stems = expected_stems()
    grids = {}
    for tol in TOLS:
        csv_path = args.csv_dir / f"results_endtoend_tol{tol}.csv"
        grids[tol] = load_status_grid(csv_path, stems, tol)
        n_csv = sum(1 for v in grids[tol].values() if v != ST_PARTIAL)
        n_partial = sum(1 for v in grids[tol].values() if v == ST_PARTIAL)
        print(f"loaded {csv_path}  ({n_csv} finished, {n_partial} partial)")
    print_summary(grids, stems)
    plot_status(grids, stems, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
