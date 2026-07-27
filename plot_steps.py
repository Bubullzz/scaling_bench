#!/usr/bin/env python3
"""
plot_steps.py -- render two per-solver / per-instance views:

  1. iterations       (# PDLP steps taken to converge or hit time limit)
  2. time per 1000 iterations   (step_s / iters * 1000, in seconds)

Both plots read the same `results_endtoend_tol<tol>.csv` produced by
endtoend_build_csv.py and re-use the solver color palette from
plot_endtoend.py so all charts in `plots/` share one visual identity.

`step_s` (loop-only time, i.e. setup and presolve NOT included) is the
right numerator for a "time / iter" metric, because those two phases
happen once per run regardless of the iteration count and would dilute
the per-step rate. `iterations` is the total step count the solver
reported before exit (Optimal / TIME LIMIT / etc.).

Runs that crashed (non-zero exit code, not a TIME LIMIT) are drawn as
empty red-outline placeholders -- no meaningful step count. Rows with
missing iterations (parser could not extract the count, e.g. broken
NCCL log) also fall through to N/A. See plot_endtoend.Row for the
crash-vs-time-limit classification, which we import to stay
consistent between plots.

Usage:
    python plot_steps.py                 # default: tol=1e-4, both plots
    python plot_steps.py --tol 1e-6
    python plot_steps.py --view iters    # just the iteration count
    python plot_steps.py --view rate     # just time/1000 iters
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Re-use the Row dataclass, color palette, TIME_LIMIT constants, and
# load_rows() from plot_endtoend so this new script never disagrees
# with the main end-to-end plot about "what counts as a crash" or
# "which color is D-PDLP".
from plot_endtoend import (
    Row, load_rows, COLOR, LABEL, TIME_LIMIT_S,
    _stem, _all_instances_sorted,
)


HERE = Path(__file__).resolve().parent

# All 3 solvers together -- the two new plots don't need the pairwise
# comparison views (dpdlp-only / cuopt-base-only) because "how many
# iters" and "sec / 1000 iters" are absolute per-solver metrics, not
# head-to-head ratios. One chart per metric is enough.
SOLVERS = ["cuopt-distributed", "dpdlp", "cuopt-base"]


# ------------------------------------------------------------------
# Metric extraction
# ------------------------------------------------------------------

def iters(r: Row) -> int | None:
    """The iteration count for the run, or None if unavailable.

    We accept iterations from any status (Optimal, TIME LIMIT, ...)
    because both matter for these plots -- e.g. "TIME LIMIT reached at
    16M iters in 3600s" is a legitimate data point for both views.
    We only reject actual crashes.
    """
    if r.is_crash:
        return None
    return r.iterations


def sec_per_1000(r: Row) -> float | None:
    """Wall-clock seconds spent per 1000 solve-loop iterations.

    Uses step_s (loop time) as the numerator on purpose: setup +
    presolve are one-time overheads that don't scale with iters and
    would inflate the ratio on short runs. Falls back to (total - pre)
    when step_s is missing but total and setup are both known.

    Returns None when either the numerator or the denominator is
    missing/zero.
    """
    if r.is_crash:
        return None
    n = r.iterations
    if not n:  # None or 0
        return None
    t_step = r.loop_s  # defined in plot_endtoend.Row as "step_s (or total - pre)"
    if t_step is None or t_step <= 0:
        return None
    return (t_step / n) * 1000.0


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

def _plot_metric(rows: list[Row],
                 metric_fn,
                 ylabel: str,
                 title: str,
                 out_path: Path,
                 log_scale: bool) -> None:
    """Grouped bar plot: instances on x-axis, one bar per solver per
    instance, y = metric_fn(row).

    log_scale=True is appropriate for both `iters` (values span 5
    orders of magnitude: 500 -> 20M) and `sec_per_1000` (spans 3-4
    orders of magnitude across solvers). Linear scale would make the
    fastest instances invisible.

    Missing values (N/A or crash) render as short red-outlined
    placeholders so the reader can still see the (solver, instance)
    slot is populated but has no measurement.
    """
    rows = [r for r in rows if r.solver in SOLVERS]
    if not rows:
        print(f"!! plot_metric: no rows for {SOLVERS}", file=sys.stderr)
        return

    instances = _all_instances_sorted(rows)
    n_ins = len(instances)
    n_sol = len(SOLVERS)

    fig_w = max(10.0, 0.55 * n_ins * n_sol + 3.5)
    fig_h = 8.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x_step = 0.7
    total_group = 0.55
    bar_w = total_group / n_sol
    xs_base = [i * x_step for i in range(n_ins)]

    by_key: dict[tuple[str, str], Row] = {(r.solver, r.instance): r for r in rows}

    # Compute ymax on the successful bars only. On log scale we also
    # need a positive ymin -- pick the smallest observed value / 2 so
    # short bars are still visible.
    vals: list[float] = []
    for r in rows:
        v = metric_fn(r)
        if v is not None and v > 0:
            vals.append(float(v))
    if not vals:
        print(f"!! plot_metric: no non-zero values for {out_path.name}", file=sys.stderr)
        plt.close(fig)
        return

    y_max = max(vals) * 1.6           # extra headroom for the value labels
    y_min = min(vals) * 0.5 if log_scale else 0.0
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0, y_max)

    # Placeholder height for N/A / crash slots: 1.5x the min so it
    # peeks above the x-axis without overlapping legit tall bars.
    na_h = y_min * 1.5 if log_scale else y_max * 0.02

    for si, solver in enumerate(SOLVERS):
        color = COLOR[solver]
        offset = (si - (n_sol - 1) / 2) * bar_w
        for xi, ins in enumerate(instances):
            r = by_key.get((solver, ins))
            x = xs_base[xi] + offset

            if r is None:
                # Slot not populated (missing (solver, instance) pair).
                ax.text(x, na_h, "N/A", ha="center", va="bottom",
                        fontsize=6, color="#999999", rotation=90)
                continue
            if r.is_crash:
                ax.bar(x, na_h, width=bar_w * 0.9,
                       facecolor="none", edgecolor="#c62828",
                       linewidth=1.5, hatch="//")
                ax.text(x, na_h * 1.05, f"CRASH\n(exit={r.exit_code})",
                        ha="center", va="bottom", fontsize=6.5,
                        color="#c62828", rotation=0)
                continue

            v = metric_fn(r)
            if v is None or v <= 0:
                ax.text(x, na_h, "N/A", ha="center", va="bottom",
                        fontsize=6, color="#999999", rotation=90)
                continue

            # Distinguish converged-Optimal from time-limit bars with
            # THREE stacked cues so the difference is unmissable at a
            # glance:
            #   1. Fill hatch  : dense `xxx` cross-hatch for TL bars,
            #                    clean solid fill for Optimal.
            #   2. Alpha       : TL bars are drawn at 0.55 (faded) so
            #                    the reader instinctively reads them
            #                    as "incomplete work". Optimal bars
            #                    are opaque.
            #   3. Text prefix : Optimal bars get a "*" star + value.
            #                    TL bars get a red "TL" tag above the
            #                    numeric label. Both fit above the bar
            #                    on log scale without collision.
            # The dashed border alone was too subtle at the resolution
            # the user reads these PNGs (their comment: "so I see
            # which ones converged vs just took 1h").
            edge = "#333"
            if r.is_time_limit:
                ax.bar(x, v, width=bar_w * 0.9, color=color,
                       edgecolor=edge, linewidth=0.8, linestyle="--",
                       hatch="xxx", alpha=0.55)
            else:
                ax.bar(x, v, width=bar_w * 0.9, color=color,
                       edgecolor=edge, linewidth=0.6, linestyle="-",
                       alpha=0.9)

            # Value label above each bar. Iteration counts get
            # thousands separators; times get 1 decimal + "s".
            if ylabel.startswith("iter"):
                if v >= 1_000_000:
                    tag = f"{v/1e6:.2f}M"
                elif v >= 1000:
                    tag = f"{v/1e3:.1f}k"
                else:
                    tag = f"{int(v)}"
            else:
                if v < 0.01:
                    tag = f"{v*1000:.1f}ms"
                elif v < 1.0:
                    tag = f"{v:.3f}s"
                else:
                    tag = f"{v:.2f}s"
            # Star-prefix converged runs (clean, tight typography); the
            # symbol is muted (#666) so the numeric value stays the
            # focus. TL runs get a red "TL" line ABOVE the number so
            # the badge is separable from the value even when bars
            # cluster.
            if r.is_time_limit:
                label_txt = f"TL\n{tag}"
                label_color = "#b71c1c"
            else:
                label_txt = f"* {tag}"
                label_color = "#222"
            # In log scale we push the label a bit above the bar
            # multiplicatively so it doesn't overlap tall neighbors.
            y_label_pos = v * 1.15 if log_scale else v + y_max * 0.01
            ax.text(x, y_label_pos, label_txt, ha="center", va="bottom",
                    fontsize=6.5, color=label_color, rotation=0)

    ax.set_xticks(xs_base)
    ax.set_xticklabels([_stem(i) for i in instances],
                       rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("instance", fontsize=11)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.5)

    legend_solver = [
        mpatches.Patch(facecolor=COLOR[s], edgecolor="#333", label=LABEL[s])
        for s in SOLVERS
    ]
    # Legend now advertises the two "healthy" outcomes as separate
    # swatches so the reader can decode the plot without a second
    # look. `hatch="xxx"` + `alpha=0.55` matches how TL bars are
    # drawn above, and the "* iters" swatch mirrors the Optimal tag.
    legend_extra = [
        mpatches.Patch(facecolor="#cccccc", edgecolor="#333",
                       linewidth=0.6, label="Optimal  (* iters shown)"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="#333",
                       linewidth=0.8, linestyle="--", hatch="xxx",
                       alpha=0.55,
                       label=f"time limit ({TIME_LIMIT_S:.0f}s, TL iters shown)"),
        mpatches.Patch(facecolor="none", edgecolor="#c62828",
                       linewidth=1.5, hatch="//", label="crash"),
    ]
    ax.legend(handles=legend_solver + legend_extra,
              loc="upper left", bbox_to_anchor=(1.005, 1.0),
              fontsize=9, framealpha=0.9)

    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_iterations(rows: list[Row], out_path: Path, tol: str) -> None:
    _plot_metric(
        rows, iters,
        ylabel="iterations (steps to solve / time-limit)",
        title=f"PDLP iterations per instance  (tol={tol})",
        out_path=out_path,
        log_scale=True,
    )


def plot_step_rate(rows: list[Row], out_path: Path, tol: str) -> None:
    _plot_metric(
        rows, sec_per_1000,
        ylabel="wall-clock seconds per 1000 solve-loop iterations",
        title=f"Cost per 1000 iterations  (tol={tol}, step_s / iters * 1000)",
        out_path=out_path,
        log_scale=True,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

VIEW_TO_FN = {
    "iters": ("iterations_per_instance",   plot_iterations),
    "rate":  ("time_per_1000_iters",       plot_step_rate),
}


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tol", choices=("1e-4", "1e-6"), default="1e-4",
                   help="which results_endtoend_tol<tol>.csv to read.")
    p.add_argument("--csv", default=None,
                   help="explicit path to the CSV (default: derived from --tol).")
    p.add_argument("--out-dir", default=None,
                   help="output directory for PNGs (default: <this dir>/plots).")
    p.add_argument("--view", choices=list(VIEW_TO_FN) + ["both"], default="both",
                   help="which chart to render. 'both' (default) writes two PNGs.")
    return p.parse_args()


def main() -> int:
    args = parse_cli()
    csv_path = Path(args.csv or (HERE / f"results_endtoend_tol{args.tol}.csv"))
    if not csv_path.is_file():
        sys.exit(f"CSV not found: {csv_path}. Run endtoend_build_csv.py --tol {args.tol} first.")
    out_dir = Path(args.out_dir or (HERE / "plots"))
    rows = load_rows(csv_path)

    views = list(VIEW_TO_FN) if args.view == "both" else [args.view]
    for v in views:
        stem, fn = VIEW_TO_FN[v]
        fn(rows, out_dir / f"{stem}_tol{args.tol}.png", tol=args.tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
