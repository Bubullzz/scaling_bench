#!/usr/bin/env python3
"""
One-bar-per-instance speedup plots for the 1v1 comparisons:
  - cuOpt Distributed vs D-PDLP
  - cuOpt Distributed vs cuOpt single-GPU (cuopt-base)

Metrics:
  wall   : t(baseline) / t(cuOpt Distributed)
           Optimal + Time Limit kept (TL capped at 3600s; hatched on plot).
           Crashes excluded.
  rate   : (s/1000 iters)_baseline / (s/1000 iters)_ours
           uses step_s / iterations * 1000; Time Limit runs kept if
           step_s and iterations are both present (rate still meaningful)

Datasets:
  all           : every instance in the CSV (default)
  addendum      : Mittelmann LPfeas ADDENDUM only (datasets.ADDENDUM_STEMS)
  non-addendum  : everything else (exclude ADDENDUM_STEMS)

Usage:
    python plot_speedup_1v1.py
    python plot_speedup_1v1.py --tol 1e-6
    python plot_speedup_1v1.py --dataset addendum --metric rate
    python plot_speedup_1v1.py --dataset non-addendum --metric rate
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from datasets import ADDENDUM_STEMS


HERE = Path(__file__).resolve().parent

DISPLAY = {
    "C5_bigger_sanitized": "C5-bigger",
    "C5_baseline_sanitized": "C5-baseline",
    "Dual2_5000": "Dual2-5000",
    "ELMOD_876_10_noVEnames": "ELMOD-876",
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
    "qap-tho-150": "qap-tho-150",
    "qap-wil-100": "qap-wil-100",
    "supply-chain": "supply-chain",
    "tsp-gaia-10m": "tsp-gaia-10m",
    "design_match": "design-match",
    "zib03": "zib03",
    "zen-garden-eur-PI-28-200ts": "zen-garden-200ts",
    "model_de_20_scenarios": "PSR6-20",
    "model_de_50_scenarios": "PSR6-50",
    "model_de_100_scenarios": "PSR6-100",
}

EXCLUDE_TOYS = {"afiro_original"}

OURS = "cuopt-distributed"
WIN_C = "#76b900ff"
LOSE_C = "#8B6B4A"
TIE_C = "#5C6670"

BASELINES = {
    "dpdlp": {
        "label": "D-PDLP",
        "wall_xlabel": r"Speedup  $t_{\mathrm{D\text{-}PDLP}}\,/\,t_{\mathrm{cuOpt}}$",
        "rate_xlabel": r"Speedup  $(\mathrm{s}/1000\,\mathrm{iters})_{\mathrm{D\text{-}PDLP}}"
                       r"\,/\,(\mathrm{s}/1000\,\mathrm{iters})_{\mathrm{cuOpt}}$",
        "win_label": "cuOpt faster",
        "lose_label": "D-PDLP faster",
        "wall_outfile": "speedup_cuopt_vs_dpdlp.png",
        "rate_outfile": "speedup_1000iters_cuopt_vs_dpdlp.png",
    },
    "cuopt-base": {
        "label": "cuOpt single-GPU",
        "wall_xlabel": r"Speedup  $t_{\mathrm{single}}\,/\,t_{\mathrm{distributed}}$",
        "rate_xlabel": r"Speedup  $(\mathrm{s}/1000\,\mathrm{iters})_{\mathrm{single}}"
                       r"\,/\,(\mathrm{s}/1000\,\mathrm{iters})_{\mathrm{distributed}}$",
        "win_label": "distributed faster",
        "lose_label": "single-GPU faster",
        "wall_outfile": "speedup_cuopt_vs_base.png",
        "rate_outfile": "speedup_1000iters_cuopt_vs_base.png",
    },
}


TIME_LIMIT_S = 3600.0


@dataclass
class Pair:
    stem: str
    label: str
    t_ours: float
    t_other: float
    ours_tl: bool = False
    other_tl: bool = False

    @property
    def speedup(self) -> float:
        return self.t_other / self.t_ours

    @property
    def any_tl(self) -> bool:
        return self.ours_tl or self.other_tl


def _stem(instance: str) -> str:
    name = Path(instance).name.replace("_gurobi_presolved", "")
    for sfx in (".mps.gz", ".mps", ".lp.gz", ".lp"):
        if name.endswith(sfx):
            return name[: -len(sfx)]
    return name


def _is_optimal(status: str) -> bool:
    return "optimal" in (status or "").strip().lower()


def _is_time_limit(status: str, total_s: float | None,
                   iterations: int | None) -> bool:
    s = (status or "").strip().upper().replace(" ", "_")
    if s in ("TIME_LIMIT", "TIMELIMIT"):
        return True
    # Same fallback as plot_endtoend: near the wall with iters accumulated.
    if (total_s is not None
            and total_s >= 0.90 * TIME_LIMIT_S
            and (iterations or 0) > 0
            and not _is_optimal(status)):
        return True
    return False


def _wall_usable(status: str, exit_code: str, total_s: float | None,
                 iterations: int | None) -> bool:
    """Optimal or Time Limit with a usable total_s; exclude crashes."""
    if total_s is None or total_s <= 0:
        return False
    if _is_optimal(status) or _is_time_limit(status, total_s, iterations):
        return True
    return False


def _is_crash(status: str, exit_code: str, total_s: float | None,
              iterations: int | None) -> bool:
    """Rough crash detector aligned with plot_endtoend (non-TL failure)."""
    if _is_time_limit(status, total_s, iterations):
        return False
    if _is_optimal(status):
        return False
    try:
        code = int(exit_code) if exit_code not in ("", None) else 0
    except ValueError:
        code = -1
    if code != 0:
        return True
    # Empty status + no useful metrics → crash / OOM / parse fail
    if not status and (total_s is None or not iterations):
        return True
    return False


def _sec_per_1000(step_s: float | None, iterations: int | None) -> float | None:
    if step_s is None or step_s <= 0 or not iterations or iterations <= 0:
        return None
    return (step_s / iterations) * 1000.0


def load_pairs(csv_path: Path, other: str, metric: str,
               allow: frozenset[str] | None = None,
               exclude: frozenset[str] | None = None) -> list[Pair]:
    by: dict[str, dict[str, dict]] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            sol = r["solver"]
            if sol not in (OURS, other):
                continue
            stem = _stem(r["instance"])
            if stem in EXCLUDE_TOYS:
                continue
            if allow is not None and stem not in allow:
                continue
            if exclude is not None and stem in exclude:
                continue
            try:
                total = float(r["total_s"]) if r.get("total_s") else None
            except ValueError:
                total = None
            try:
                step = float(r["step_s"]) if r.get("step_s") else None
            except ValueError:
                step = None
            try:
                iters = int(float(r["iterations"])) if r.get("iterations") else None
            except ValueError:
                iters = None
            by.setdefault(stem, {})[sol] = {
                "total": total,
                "step": step,
                "iters": iters,
                "status": r.get("status") or "",
                "exit_code": r.get("exit_code") or "",
            }

    pairs: list[Pair] = []
    for stem, d in by.items():
        a, b = d.get(OURS), d.get(other)
        if not a or not b:
            continue
        if metric == "wall":
            if not (_wall_usable(a["status"], a["exit_code"], a["total"], a["iters"])
                    and _wall_usable(b["status"], b["exit_code"], b["total"], b["iters"])):
                continue
            # Cap at the wall so a TL run that slightly overshoots (e.g. 3605s)
            # doesn't invent extra speedup over another TL at 3600s.
            t_ours = min(a["total"], TIME_LIMIT_S)
            t_other = min(b["total"], TIME_LIMIT_S)
            pairs.append(Pair(
                stem, DISPLAY.get(stem, stem), t_ours, t_other,
                ours_tl=_is_time_limit(a["status"], a["total"], a["iters"]),
                other_tl=_is_time_limit(b["status"], b["total"], b["iters"]),
            ))
        else:  # rate
            if _is_crash(a["status"], a["exit_code"], a["total"], a["iters"]):
                continue
            if _is_crash(b["status"], b["exit_code"], b["total"], b["iters"]):
                continue
            ra = _sec_per_1000(a["step"], a["iters"])
            rb = _sec_per_1000(b["step"], b["iters"])
            if ra is None or rb is None:
                continue
            pairs.append(Pair(stem, DISPLAY.get(stem, stem), ra, rb))

    pairs.sort(key=lambda p: p.speedup)
    return pairs


def _tl_suffix(p: Pair, other_short: str = "base") -> str:
    if p.ours_tl and p.other_tl:
        return "  [both TL]"
    if p.ours_tl:
        return "  [cuOpt TL]"
    if p.other_tl:
        return f"  [{other_short} TL]"
    return ""


def _speedup_tag(p: Pair) -> str:
    """Bar annotation. Baseline-only TL is a lower bound on speedup
    (they were capped at 3600s), so show '>n.nx' rather than 'n.nx*'."""
    s = p.speedup
    if p.other_tl and not p.ours_tl:
        return f">{s:.2f}×"
    if p.any_tl:
        return f"{s:.2f}×*"
    return f"{s:.2f}×"


def plot_speedup(pairs: list[Pair], cfg: dict, out_path: Path,
                 xlabel: str) -> None:
    if not pairs:
        print(f"!! no pairs for {out_path.name}", file=sys.stderr)
        return

    # Short TL tag for the baseline: "base" for single-GPU, "D-PDLP" otherwise.
    other_short = "base" if "single" in cfg["label"].lower() else cfg["label"]

    n = len(pairs)
    speedups = [p.speedup for p in pairs]
    labels = [p.label + _tl_suffix(p, other_short) for p in pairs]
    wins = sum(1 for s in speedups if s > 1.01)
    losses = sum(1 for s in speedups if s < 0.99)
    geomean = math.exp(sum(math.log(s) for s in speedups) / n)
    colors = [WIN_C if s > 1.01 else (LOSE_C if s < 0.99 else TIE_C) for s in speedups]
    has_tl = any(p.any_tl for p in pairs)

    fig_h = max(4.8, 0.38 * n + 1.4)
    fig, ax = plt.subplots(figsize=(8.8, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = range(n)
    bars = ax.barh(y, speedups, color=colors, height=0.72, edgecolor="none", zorder=2)
    # Hatch any bar involving a time-limit so TL cases are visually distinct.
    for bar, p in zip(bars, pairs):
        if p.any_tl:
            bar.set_hatch("////")
            bar.set_edgecolor("#1A1A1A")
            bar.set_linewidth(0.4)

    ax.axvline(1.0, color="#1A1A1A", linewidth=1.2, zorder=3)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11)
    # Read the ranking from top to bottom: lowest speedup first and strongest
    # speedup last at the bottom.
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xlim(0, max(2.8, math.ceil(max(speedups) * 10) / 10 + 0.2))

    for i, p in enumerate(pairs):
        ax.text(p.speedup + 0.04, i, _speedup_tag(p), va="center", ha="left",
                fontsize=9.5, color="#1A1A1A")

    # Keep the geomean outside the data area so it never covers a bar.
    ax.set_title(f"Geometric mean speedup: {geomean:.2f}×",
                 loc="left", fontsize=11, pad=10, color="#1A1A1A")

    handles = [
        Patch(facecolor=WIN_C, edgecolor="none", label=cfg["win_label"]),
        Patch(facecolor=LOSE_C, edgecolor="none", label=cfg["lose_label"]),
        plt.Line2D([0], [0], color="#1A1A1A", linewidth=1.2, label="parity (1×)"),
    ]
    if has_tl:
        # Use a real Patch so the hatch renders in the legend (empty barh
        # handles often show as a flat colour and look "wrong").
        handles.append(Patch(
            facecolor="#E8E8E8", edgecolor="#1A1A1A", linewidth=0.6,
            hatch="////", label="involves time limit",
        ))
    leg = ax.legend(handles=handles, loc="upper right", frameon=True,
                    fontsize=11, fancybox=False, edgecolor="#CCCCCC",
                    framealpha=1.0)
    leg.get_frame().set_linewidth(0.6)

    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  n={n}  wins={wins}  losses={losses}  geomean={geomean:.3f}×"
          f"  (tl_pairs={sum(1 for p in pairs if p.any_tl)})")
    for p in pairs:
        print(f"  {p.label + _tl_suffix(p, other_short):40s}  {_speedup_tag(p):>8s}  "
              f"(ours={p.t_ours:.4g}  other={p.t_other:.4g})")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tol", choices=("1e-4", "1e-6"), default="1e-6")
    p.add_argument("--csv", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--vs", choices=("dpdlp", "cuopt-base", "both"), default="both")
    p.add_argument("--metric", choices=("wall", "rate"), default="wall",
                   help="wall = total_s speedup; rate = s/1000-iters speedup")
    p.add_argument("--dataset", choices=("all", "addendum", "non-addendum"),
                   default="all")
    return p.parse_args()


def main() -> int:
    args = parse_cli()
    csv_path = Path(args.csv or (HERE / f"results_endtoend_tol{args.tol}.csv"))
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    allow = ADDENDUM_STEMS if args.dataset == "addendum" else None
    exclude = ADDENDUM_STEMS if args.dataset == "non-addendum" else None
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.dataset in ("addendum", "non-addendum"):
        out_dir = HERE / "plots" / args.tol / args.dataset
    else:
        out_dir = HERE / "plots" / args.tol

    metric_key = "rate" if args.metric == "rate" else "wall"
    targets = list(BASELINES) if args.vs == "both" else [args.vs]
    for other in targets:
        cfg = BASELINES[other]
        pairs = load_pairs(csv_path, other, args.metric, allow=allow, exclude=exclude)
        outfile = cfg[f"{metric_key}_outfile"]
        xlabel = cfg[f"{metric_key}_xlabel"]
        plot_speedup(pairs, cfg, out_dir / outfile, xlabel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
