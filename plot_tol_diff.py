#!/usr/bin/env python3
"""Compare two solvers' outcomes at both tolerances.

The plot includes only instances where exactly one of the two solvers is
Optimal at 1e-4 and/or 1e-6. Four columns show both solvers at both
tolerances, with wall-clock time in each cell.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from plot_run_status import (
    ST_CRASH,
    ST_COLOR,
    ST_HANG,
    ST_LABEL,
    ST_MISSING,
    ST_OPTIMAL,
    ST_PARTIAL,
    ST_TL,
    TOLS,
    _as_float,
    _display,
    _stem,
    expected_stems,
    load_status_grid,
)


HERE = Path(__file__).resolve().parent

SOLVER_LABEL = {
    "cuopt-distributed": "cuOpt-dist",
    "cuopt-base": "cuOpt-base",
    "dpdlp": "D-PDLP",
}
SOLVER_SHORT = {
    "cuopt-distributed": "cuOpt-dist",
    "cuopt-base": "cuOpt-base",
    "dpdlp": "D-PDLP",
}


def load_times(
    csv_path: Path,
    solvers: tuple[str, str],
) -> dict[tuple[str, str], float | None]:
    """Return {(solver, stem): total_s} from a results CSV."""
    want = set(solvers)
    out: dict[tuple[str, str], float | None] = {}
    if not csv_path.is_file():
        return out
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            solver = (row.get("solver") or "").strip()
            if solver not in want:
                continue
            stem = _stem(row.get("instance", ""))
            out[(solver, stem)] = _as_float(row.get("total_s"))
    return out


def collect_instances(
    grids: dict[str, dict[tuple[str, str], int]],
    stems: list[str],
    solvers: tuple[str, str],
) -> list[str]:
    """Instances where one of the two solvers is Optimal and the other is not."""
    a, b = solvers
    selected: list[str] = []
    for stem in stems:
        differs = any(
            (grids[tol].get((a, stem), ST_MISSING) == ST_OPTIMAL)
            != (grids[tol].get((b, stem), ST_MISSING) == ST_OPTIMAL)
            for tol in TOLS
        )
        if differs:
            selected.append(stem)

    def sort_key(stem: str) -> tuple[int, str]:
        n_diff = sum(
            (grids[tol].get((a, stem), ST_MISSING) == ST_OPTIMAL)
            != (grids[tol].get((b, stem), ST_MISSING) == ST_OPTIMAL)
            for tol in TOLS
        )
        return (-n_diff, _display(stem).lower())

    return sorted(selected, key=sort_key)


def _fmt_time(t: float | None, code: int) -> str:
    if code in (ST_MISSING, ST_PARTIAL, ST_HANG):
        return ""
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return ""
    if t >= 100:
        return f"{t:.0f}s"
    if t >= 10:
        return f"{t:.1f}s"
    return f"{t:.2f}s"


def plot_comparison(
    grids: dict[str, dict[tuple[str, str], int]],
    times: dict[str, dict[tuple[str, str], float | None]],
    stems: list[str],
    solvers: tuple[str, str],
    out_path: Path,
) -> list[str]:
    a, b = solvers
    selected = collect_instances(grids, stems, solvers)
    n = max(len(selected), 1)
    fig, ax = plt.subplots(figsize=(12.5, max(6.0, 0.55 * n + 2.6)))

    columns = [
        ("1e-4", a, f"{SOLVER_LABEL[a]}\n1e-4"),
        ("1e-4", b, f"{SOLVER_LABEL[b]}\n1e-4"),
        ("1e-6", a, f"{SOLVER_LABEL[a]}\n1e-6"),
        ("1e-6", b, f"{SOLVER_LABEL[b]}\n1e-6"),
    ]

    if not selected:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"No instance has exactly one of {{{a}, {b}}} Optimal",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
    else:
        matrix = [
            [grids[tol].get((solver, stem), ST_MISSING) for tol, solver, _ in columns]
            for stem in selected
        ]
        cmap = ListedColormap([ST_COLOR[i] for i in range(6)])
        norm = BoundaryNorm(
            [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            cmap.N,
        )
        ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([label for _, _, label in columns], fontsize=10)
        ax.set_yticks(range(len(selected)))
        ax.set_yticklabels([_display(stem) for stem in selected], fontsize=8.5)

        ax.set_xticks([x - 0.5 for x in range(len(columns) + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(len(selected) + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Strong divider between tolerance groups.
        ax.axvline(1.5, color="#222222", linewidth=2.0)

        marks = {
            ST_OPTIMAL: "O",
            ST_TL: "T",
            ST_CRASH: "X",
            ST_HANG: "H",
            ST_PARTIAL: "P",
            ST_MISSING: "·",
        }
        for row, stem in enumerate(selected):
            for col, (tol, solver, _) in enumerate(columns):
                code = matrix[row][col]
                t = times[tol].get((solver, stem))
                time_s = _fmt_time(t, code)
                label = marks[code] if not time_s else f"{marks[code]}\n{time_s}"
                dark_text = code in (ST_TL, ST_PARTIAL, ST_MISSING)
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color="#222222" if dark_text else "white",
                    linespacing=1.05,
                )

    handles = [
        mpatches.Patch(color=ST_COLOR[code], label=ST_LABEL[code])
        for code in (
            ST_OPTIMAL,
            ST_TL,
            ST_CRASH,
            ST_HANG,
            ST_PARTIAL,
            ST_MISSING,
        )
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.60, 0.98),
    )
    fig.suptitle(
        f"{SOLVER_LABEL[a]} vs {SOLVER_LABEL[b]}: "
        "instances where only one solver is Optimal",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    ax.set_title(
        "Both tolerances  |  O=Optimal, T=TL, X=Crash, H=Hang  |  times = total_s",
        fontsize=10.5,
        pad=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return selected


def print_instances(
    grids: dict[str, dict[tuple[str, str], int]],
    times: dict[str, dict[tuple[str, str], float | None]],
    selected: list[str],
    solvers: tuple[str, str],
) -> None:
    a, b = solvers
    print(
        f"\n=== {len(selected)} instances: exactly one of "
        f"{{{a}, {b}}} Optimal ==="
    )
    for stem in selected:
        cells = []
        for tol in TOLS:
            for solver in solvers:
                code = grids[tol].get((solver, stem), ST_MISSING)
                t = times[tol].get((solver, stem))
                ts = _fmt_time(t, code) or "—"
                cells.append(
                    f"{SOLVER_SHORT[solver]}@{tol}={ST_LABEL[code]} ({ts})"
                )
        print(f"  {_display(stem):28s}")
        print(f"    {'; '.join(cells)}")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vs",
        choices=("dpdlp", "cuopt-base"),
        default="dpdlp",
        help="compare cuopt-distributed against this solver (default: dpdlp)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PNG (default: plots/tol_diff.png or plots/tol_diff_vs_base.png)",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=HERE,
        help="directory containing results_endtoend_tol*.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_cli()
    solvers = ("cuopt-distributed", args.vs)
    out_path = args.out
    if out_path is None:
        name = (
            "tol_diff.png"
            if args.vs == "dpdlp"
            else "tol_diff_vs_base.png"
        )
        out_path = HERE / "plots" / name

    stems = expected_stems()
    grids: dict[str, dict[tuple[str, str], int]] = {}
    times: dict[str, dict[tuple[str, str], float | None]] = {}
    for tol in TOLS:
        csv_path = args.csv_dir / f"results_endtoend_tol{tol}.csv"
        grids[tol] = load_status_grid(csv_path, stems, tol)
        times[tol] = load_times(csv_path, solvers)
        print(f"loaded {csv_path}")
    selected = plot_comparison(grids, times, stems, solvers, out_path)
    print_instances(grids, times, selected, solvers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
