#!/usr/bin/env python3
"""Scatter: problem size (nnz) vs speedup — companion to plot_speedup_1v1.py.

One point per instance. X = nonzeros (log scale, from cuOpt logs),
Y = speedup vs the chosen baseline (same pairing rules as the bar charts).

Usage:
    python plot_speedup_vs_nnz.py --tol 1e-6 --dataset all
    python plot_speedup_vs_nnz.py --tol 1e-4 --dataset addendum --vs dpdlp
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from datasets import NAMED_DATASET_NAMES, dataset_filter, plot_out_dir, validate_datasets
from plot_speedup_1v1 import (
    BASELINES,
    HERE,
    OURS,
    WIN_C,
    LOSE_C,
    TIE_C,
    _fmt_sgm_time,
    failure_sidebar_lines,
    load_outcomes,
    load_pairs,
    shifted_geomean,
)


_DIMS = re.compile(
    r"Solving a problem with\s+(?P<n_cstr>\d+)\s+constraints,\s+"
    r"(?P<n_vars>\d+)\s+variables\s+\((?P<n_int>\d+)\s+integers\),\s+and\s+"
    r"(?P<nnz>\d+)\s+nonzeros"
)


def _stem_from_log_name(name: str) -> str:
    stem = name.split("__N")[0]
    for suf in ("_PSLP_presolved", "_gurobi_presolved"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    if stem == "psr-100":
        stem = "psr_100"
    return stem


def load_nnz_by_stem(runs_root: Path) -> dict[str, int]:
    """Prefer cuopt-distributed logs, fall back to cuopt-base."""
    out: dict[str, int] = {}
    for sol in ("cuopt-distributed", "cuopt-base"):
        for log in runs_root.glob(f"*/{sol}/*.log"):
            stem = _stem_from_log_name(log.name)
            if stem in out:
                continue
            try:
                text = log.read_text(errors="ignore")
            except OSError:
                continue
            m = _DIMS.search(text)
            if m:
                out[stem] = int(m.group("nnz"))
    return out


def _wrap_ann_label(label: str, width: int = 22) -> str:
    """Wrap annotation labels without mid-token splits (no '100t\\ns')."""
    if len(label) <= width:
        return label
    lines = textwrap.wrap(
        label,
        width=width,
        break_long_words=False,
        break_on_hyphens=True,
    )
    return "\n".join(lines) if lines else label


def _annotation_offsets(
    items: list[tuple],
) -> list[tuple[float, float]]:
    """Pick non-colliding label offsets; does not change data/axes.

    Spatially nearby points get consecutive fan slots (above/below/left).
    Right-side points bias left so labels stay out of the sidebar.
    """
    if not items:
        return []
    fan = [
        (8, 10), (8, -14), (-60, 8), (-60, -12),
        (12, 2), (-75, 2), (8, 22), (8, -24),
        (-45, 18), (-45, -20), (18, 8), (-90, 10),
        (8, 32), (-60, 28), (20, -8), (-80, -18),
    ]
    xs = [nnz for _, nnz in items]
    xmax = max(xs)
    # Order by (speedup, log nnz) so neighbors get different fan slots.
    order = sorted(
        range(len(items)),
        key=lambda i: (items[i][0].speedup, math.log10(max(items[i][1], 1.0))),
    )
    slot_of = {i: k for k, i in enumerate(order)}
    out: list[tuple[float, float]] = []
    for i, (_, nnz) in enumerate(items):
        dx, dy = fan[slot_of[i] % len(fan)]
        if nnz >= xmax / 4:
            dx = -max(abs(dx), 60)
        out.append((dx, dy))
    return out


def plot_scatter(pairs, nnz_by_stem: dict[str, int], cfg: dict,
                 out_path: Path, title: str, ylabel: str,
                 *, value_unit: str = "s",
                 outcomes: dict[str, dict[str, str]] | None = None,
                 other_solver: str = "dpdlp") -> None:
    # Same policy as bar charts: drop both-TL artificial 1× points.
    both_tl = [p for p in pairs if p.ours_tl and p.other_tl]
    pairs = [p for p in pairs if not (p.ours_tl and p.other_tl)]
    pts = []
    missing = []
    for p in pairs:
        nnz = nnz_by_stem.get(p.stem)
        if nnz is None or nnz <= 0:
            missing.append(p.stem)
            continue
        pts.append((p, nnz))
    fail_lines = failure_sidebar_lines(outcomes or {}, (OURS, other_solver))
    if not pts:
        print(f"!! no nnz-annotated pairs for {out_path.name}", file=sys.stderr)
        if missing:
            print(f"  missing nnz: {', '.join(missing)}", file=sys.stderr)
        if not fail_lines:
            return

    if pts:
        xs = [nnz for _, nnz in pts]
        ys = [p.speedup for p, _ in pts]
        colors = [
            WIN_C if p.speedup > 1.01 else (LOSE_C if p.speedup < 0.99 else TIE_C)
            for p, _ in pts
        ]
        markers = ["D" if p.any_tl else "o" for p, _ in pts]
    else:
        xs = ys = colors = markers = []

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    if pts:
        for (p, nnz), c, m in zip(pts, colors, markers):
            ax.scatter(
                nnz, p.speedup, s=78, c=c, marker=m,
                edgecolors="#1A1A1A", linewidths=0.45, zorder=3, alpha=0.92,
            )

        ax.axhline(1.0, color="#1A1A1A", linewidth=1.1, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("Number of nonzeros (nnz)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, max(2.2, math.ceil(max(ys) * 10) / 10 + 0.15))
        xmin = min(xs) / 1.4
        xmax = max(xs) * 1.4
        ax.set_xlim(xmin, xmax)

        ranked_spd = sorted(pts, key=lambda t: -t[0].speedup)
        ranked_nnz = sorted(pts, key=lambda t: -t[1])
        annotate = {}
        for p, nnz in ranked_spd[:5] + ranked_nnz[:5]:
            annotate[p.stem] = (p, nnz)
        for p, nnz in sorted(pts, key=lambda t: t[0].speedup)[:3]:
            annotate[p.stem] = (p, nnz)

        ann_items = list(annotate.values())
        offsets = _annotation_offsets(ann_items)
        for (p, nnz), (dx, dy) in zip(ann_items, offsets):
            ax.annotate(
                _wrap_ann_label(p.label),
                (nnz, p.speedup),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=6.5,
                color="#333333",
                alpha=0.95,
                ha="right" if dx < 0 else "left",
                va="bottom" if dy >= 0 else "top",
                clip_on=True,
                linespacing=0.95,
            )

        geomean = math.exp(sum(math.log(y) for y in ys) / len(ys))
        sgm_ours = shifted_geomean([p.t_ours for p, _ in pts])
        sgm_other = shifted_geomean([p.t_other for p, _ in pts])
        other_short = cfg["label"]
        ax.set_title(
            f"n={len(pts)}  |  geometric mean speedup: {geomean:.2f}×  |  "
            f"SGM10 times: cuOpt_distributed="
            f"{_fmt_sgm_time(sgm_ours, value_unit)}, {other_short}="
            f"{_fmt_sgm_time(sgm_other, value_unit)}",
            loc="left", fontsize=10, pad=8, color="#1A1A1A",
        )
    else:
        ax.set_axis_off()
        other_short = cfg["label"]
        geomean = float("nan")
        sgm_ours = sgm_other = float("nan")
        ax.set_title("No comparable points (see TL/crash notes)",
                     loc="left", fontsize=11, pad=8, color="#1A1A1A")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=WIN_C,
               markeredgecolor="#1A1A1A", markersize=9,
               label=cfg["win_label"]),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LOSE_C,
               markeredgecolor="#1A1A1A", markersize=9,
               label=cfg["lose_label"]),
        Line2D([0], [0], color="#1A1A1A", linewidth=1.1, label="parity (1×)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E8E8E8",
               markeredgecolor="#1A1A1A", markersize=8,
               label="involves time limit"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=True, fontsize=10,
              fancybox=False, edgecolor="#CCCCCC", framealpha=1.0)

    note_blocks: list[str] = []
    if missing:
        note_blocks.append(
            "Missing nnz:\n"
            + "\n".join(f"• {s}" for s in missing[:12])
        )
    if fail_lines:
        note_blocks.append(
            "Both solvers TL or crash\n(not plotted):\n"
            + "\n".join(fail_lines)
        )
    if note_blocks:
        # Sit below the legend so labels near the top-right don't collide.
        ax.text(
            1.02, 0.42 if pts else 0.95,
            "\n\n".join(note_blocks),
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            color="#1A1A1A", clip_on=False, family="monospace",
            linespacing=1.2,
        )

    if pts:
        ax.grid(True, which="both", color="#E6E6E6", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout(rect=(0, 0, 0.78, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")
    if pts:
        print(f"  n={len(pts)}  missing_nnz={len(missing)}  "
              f"geomean={geomean:.3f}×  "
              f"sgm10_ours={sgm_ours:.3g}{value_unit}  "
              f"sgm10_{other_short}={sgm_other:.3g}{value_unit}  "
              f"both_fail_notes={len(fail_lines)}")
    if fail_lines:
        print("  both-solver TL/crash notes:")
        for line in fail_lines:
            print(f"    {line}")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tol", choices=("1e-4", "1e-6"), default="1e-6")
    p.add_argument("--csv", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--vs", choices=("dpdlp", "cuopt-base", "both"), default="both")
    p.add_argument("--metric", choices=("wall", "rate"), default="wall")
    p.add_argument(
        "--dataset",
        choices=("non-addendum", *NAMED_DATASET_NAMES),
        default="all",
    )
    return p.parse_args()


def main() -> int:
    args = parse_cli()
    validate_datasets()
    csv_path = Path(args.csv or (HERE / f"results_endtoend_tol{args.tol}.csv"))
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    nnz_by_stem = load_nnz_by_stem(HERE / "runs_endtoend")
    allow, exclude = dataset_filter(args.dataset)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else plot_out_dir(HERE / "plots", args.tol, args.dataset)
    )

    metric_key = "rate" if args.metric == "rate" else "wall"
    metric_title = (
        "1000-iteration speedup vs nnz" if args.metric == "rate"
        else "Wall-clock speedup vs nnz"
    )
    targets = list(BASELINES) if args.vs == "both" else [args.vs]
    outcomes = load_outcomes(csv_path, allow=allow, exclude=exclude)
    for other in targets:
        cfg = BASELINES[other]
        pairs = load_pairs(csv_path, other, args.metric, allow=allow, exclude=exclude)
        outfile = (
            f"speedup_vs_nnz_{'1000iters_' if args.metric == 'rate' else ''}"
            f"cuopt_vs_{'dpdlp' if other == 'dpdlp' else 'base'}.png"
        )
        ylabel = cfg[f"{metric_key}_xlabel"]
        title = (
            f"{metric_title}: {cfg['comparison_title']}\n"
            f"Dataset: {args.dataset}  |  tolerance: {args.tol}"
            "  |  time limit: 1 h"
        )
        plot_scatter(
            pairs, nnz_by_stem, cfg, out_dir / outfile, title, ylabel,
            value_unit=("s/1k" if args.metric == "rate" else "s"),
            outcomes=outcomes,
            other_solver=other,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
