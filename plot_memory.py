#!/usr/bin/env python3
"""
GPU memory-footprint plots: cuOpt Distributed vs D-PDLP (8×GPU).

Reads results_endtoend_tol<tol>.csv and plots peak used memory on the
hottest GPU (gpu_peak_mb from the NVML sampler) — the OOM-relevant
per-device number, not the sum across GPUs.

Usage:
    python plot_memory.py
    python plot_memory.py --tol 1e-6
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
    "zib03": "zib03",
    "zen-garden-eur-PI-28-200ts": "zen-garden-200ts",
    "model_de_20_scenarios": "PSR6-20",
    "model_de_50_scenarios": "PSR6-50",
    "model_de_100_scenarios": "PSR6-100",
}

# Toy LP + early-crash stubs with tiny bogus peaks.
EXCLUDE = {
    "afiro_original",
    "BEAM_4032_11_8_CLI",
    "world-shipping",
    "design_match",
}

CUOPT_C = "#76b900"   # NVIDIA green (distributed)
BASE_C = "#A8D08D"    # lighter green (single-GPU baseline)
DPDLP_C = "#5C6670"   # slate



@dataclass
class MemPair:
    stem: str
    label: str
    peak_cuopt_mb: float
    peak_dpdlp_mb: float

    @property
    def cuopt_gb(self) -> float:
        return self.peak_cuopt_mb / 1024.0

    @property
    def dpdlp_gb(self) -> float:
        return self.peak_dpdlp_mb / 1024.0


def _stem(instance: str) -> str:
    name = Path(instance).name
    for suf in ("_PSLP_presolved", "_gurobi_presolved"):
        name = name.replace(suf, "")
    for sfx in (".mps.gz", ".mps", ".lp.gz", ".lp"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]
            break
    if name == "psr-100":
        name = "psr_100"
    return name


def load_pairs(csv_path: Path) -> list[MemPair]:
    by: dict[str, dict[str, float]] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            sol = r["solver"]
            if sol not in ("cuopt-distributed", "dpdlp"):
                continue
            stem = _stem(r["instance"])
            if stem in EXCLUDE:
                continue
            try:
                peak = float(r["gpu_peak_mb"])
            except (TypeError, ValueError):
                continue
            if peak <= 0:
                continue
            by.setdefault(stem, {})[sol] = peak

    pairs: list[MemPair] = []
    for stem, d in by.items():
        if "cuopt-distributed" not in d or "dpdlp" not in d:
            continue
        # Drop early-abort stubs (sampler barely moved).
        if d["cuopt-distributed"] < 2048 or d["dpdlp"] < 2048:
            continue
        pairs.append(MemPair(
            stem=stem,
            label=DISPLAY.get(stem, stem),
            peak_cuopt_mb=d["cuopt-distributed"],
            peak_dpdlp_mb=d["dpdlp"],
        ))
    # Largest cuOpt peak first (top of barh after reverse sort via y).
    pairs.sort(key=lambda p: p.peak_cuopt_mb)
    return pairs


def plot_grouped(pairs: list[MemPair], out_path: Path, tol: str) -> None:
    if not pairs:
        sys.exit(f"no memory pairs for tol={tol}")

    n = len(pairs)
    y = list(range(n))
    h = 0.38
    cu = [p.cuopt_gb for p in pairs]
    dp = [p.dpdlp_gb for p in pairs]
    labels = [p.label for p in pairs]

    ratios = [p.peak_cuopt_mb / p.peak_dpdlp_mb for p in pairs]
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))

    fig_h = max(5.0, 0.40 * n + 1.2)
    fig, ax = plt.subplots(figsize=(9.2, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh([yi - h / 2 for yi in y], cu, height=h, color=CUOPT_C,
            edgecolor="none", zorder=2, label="cuOpt Distributed")
    ax.barh([yi + h / 2 for yi in y], dp, height=h, color=DPDLP_C,
            edgecolor="none", zorder=2, label="D-PDLP")

    xmax = max(max(cu), max(dp))
    ax.set_xlim(0, xmax * 1.18)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Peak GPU memory (GB, hottest device)", fontsize=12)

    for i, (c, d) in enumerate(zip(cu, dp)):
        ax.text(c + xmax * 0.01, i - h / 2, f"{c:.1f}", va="center",
                ha="left", fontsize=8, color="#222222")
        ax.text(d + xmax * 0.01, i + h / 2, f"{d:.1f}", va="center",
                ha="left", fontsize=8, color="#222222")

    leg = ax.legend(loc="lower right", frameon=True, fontsize=13,
                    fancybox=False, edgecolor="#CCCCCC", framealpha=1.0)
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
    print(f"  n={n}  geomean(cuOpt/D-PDLP peak)={gm:.2f}×  "
          f"(>1 ⇒ cuOpt uses more per-GPU memory)")


def plot_ratio(pairs: list[MemPair], out_path: Path, tol: str) -> None:
    """Memory ratio cuOpt/D-PDLP — honest companion to the speedup poster."""
    del tol
    if not pairs:
        return
    n = len(pairs)
    ratios = [p.peak_cuopt_mb / p.peak_dpdlp_mb for p in pairs]
    labels = [p.label for p in pairs]
    # sort by ratio ascending
    order = sorted(range(n), key=lambda i: ratios[i])
    ratios = [ratios[i] for i in order]
    labels = [labels[i] for i in order]
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))

    # Higher ratio = cuOpt hungrier (not a "win") — use slate for >1, green only if cuOpt leaner
    colors = [CUOPT_C if r < 0.99 else DPDLP_C for r in ratios]

    fig_h = max(4.8, 0.38 * n + 1.2)
    fig, ax = plt.subplots(figsize=(8.8, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = range(n)
    ax.barh(y, ratios, color=colors, height=0.72, edgecolor="none", zorder=2)
    ax.axvline(1.0, color="#1A1A1A", linewidth=1.2, zorder=3)
    ax.axvline(gm, color="#1A1A1A", linewidth=1.0, linestyle=(0, (3, 2)),
               alpha=0.55, zorder=3)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(r"Memory ratio  $\mathrm{peak}_{\mathrm{cuOpt}}\,/\,\mathrm{peak}_{\mathrm{D\text{-}PDLP}}$",
                  fontsize=12)
    ax.set_xlim(0, max(3.4, math.ceil(max(ratios) * 10) / 10 + 0.15))

    for i, r in enumerate(ratios):
        ax.text(r + 0.04, i, f"{r:.2f}×", va="center", ha="left", fontsize=9.5)

    ax.plot([], [], color=DPDLP_C, linewidth=8, label="cuOpt uses more")
    ax.plot([], [], color=CUOPT_C, linewidth=8, label="cuOpt uses less")
    ax.plot([], [], color="#1A1A1A", linewidth=1.2, label="parity (1×)")
    ax.plot([], [], color="#1A1A1A", linewidth=1.0, linestyle=(0, (3, 2)),
            label=f"geomean ({gm:.2f}×)")
    leg = ax.legend(loc="lower right", frameon=True, fontsize=13,
                    fancybox=False, edgecolor="#CCCCCC", framealpha=1.0)
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


def load_dist_vs_base(csv_path: Path) -> list[MemPair]:
    """Reuse MemPair with peak_cuopt=distributed, peak_dpdlp field = base.

    (Field name is historical; values are interpreted by the dist-vs-base
    plotters below.)
    """
    by: dict[str, dict[str, float]] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            sol = r["solver"]
            if sol not in ("cuopt-distributed", "cuopt-base"):
                continue
            stem = _stem(r["instance"])
            if stem in EXCLUDE:
                continue
            try:
                peak = float(r["gpu_peak_mb"])
            except (TypeError, ValueError):
                continue
            if peak <= 0:
                continue
            by.setdefault(stem, {})[sol] = peak

    pairs: list[MemPair] = []
    for stem, d in by.items():
        if "cuopt-distributed" not in d or "cuopt-base" not in d:
            continue
        if d["cuopt-distributed"] < 2048 or d["cuopt-base"] < 2048:
            continue
        pairs.append(MemPair(
            stem=stem,
            label=DISPLAY.get(stem, stem),
            peak_cuopt_mb=d["cuopt-distributed"],
            peak_dpdlp_mb=d["cuopt-base"],  # reused slot = single-GPU peak
        ))
    pairs.sort(key=lambda p: p.peak_dpdlp_mb)  # sort by single-GPU peak
    return pairs


def plot_dist_vs_base(pairs: list[MemPair], out_path: Path, tol: str) -> None:
    """Grouped bars: 8-GPU distributed peak/device vs 1-GPU cuOpt peak."""
    del tol
    if not pairs:
        sys.exit("no dist-vs-base memory pairs")

    n = len(pairs)
    y = list(range(n))
    h = 0.38
    dist = [p.cuopt_gb for p in pairs]          # distributed hottest GPU
    base = [p.dpdlp_gb for p in pairs]          # single-GPU peak (reused field)
    labels = [p.label for p in pairs]

    ratios = [p.peak_dpdlp_mb / p.peak_cuopt_mb for p in pairs]  # base/dist
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))

    fig_h = max(5.0, 0.40 * n + 1.2)
    fig, ax = plt.subplots(figsize=(9.2, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh([yi - h / 2 for yi in y], dist, height=h, color=CUOPT_C,
            edgecolor="none", zorder=2, label="cuOpt Distributed (8 GPU, hottest)")
    ax.barh([yi + h / 2 for yi in y], base, height=h, color=BASE_C,
            edgecolor="none", zorder=2, label="cuOpt single-GPU")

    xmax = max(max(dist), max(base))
    ax.set_xlim(0, xmax * 1.14)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Peak GPU memory (GB)", fontsize=12)

    for i, (d, b) in enumerate(zip(dist, base)):
        ax.text(d + xmax * 0.008, i - h / 2, f"{d:.1f}", va="center",
                ha="left", fontsize=8, color="#222222")
        ax.text(b + xmax * 0.008, i + h / 2, f"{b:.1f}", va="center",
                ha="left", fontsize=8, color="#222222")

    leg = ax.legend(loc="lower right", frameon=True, fontsize=12,
                    fancybox=False, edgecolor="#CCCCCC", framealpha=1.0)
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
    print(f"  n={n}  geomean(single/distributed peak)={gm:.2f}×  "
          f"(>1 ⇒ single-GPU uses more on one device)")


def plot_dist_vs_base_ratio(pairs: list[MemPair], out_path: Path, tol: str) -> None:
    del tol
    if not pairs:
        return
    n = len(pairs)
    # ratio = single / distributed  (>1 ⇒ distribution reduces per-device peak)
    ratios = [p.peak_dpdlp_mb / p.peak_cuopt_mb for p in pairs]
    labels = [p.label for p in pairs]
    order = sorted(range(n), key=lambda i: ratios[i])
    ratios = [ratios[i] for i in order]
    labels = [labels[i] for i in order]
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))

    colors = [CUOPT_C if r > 1.01 else BASE_C for r in ratios]

    fig_h = max(4.8, 0.38 * n + 1.2)
    fig, ax = plt.subplots(figsize=(8.8, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = range(n)
    ax.barh(y, ratios, color=colors, height=0.72, edgecolor="none", zorder=2)
    ax.axvline(1.0, color="#1A1A1A", linewidth=1.2, zorder=3)
    ax.axvline(gm, color="#1A1A1A", linewidth=1.0, linestyle=(0, (3, 2)),
               alpha=0.55, zorder=3)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(
        r"Memory ratio  $\mathrm{peak}_{\mathrm{single}}\,/\,\mathrm{peak}_{\mathrm{distributed}}$",
        fontsize=12,
    )
    ax.set_xlim(0, max(6.2, math.ceil(max(ratios) * 10) / 10 + 0.2))

    for i, r in enumerate(ratios):
        ax.text(r + 0.06, i, f"{r:.2f}×", va="center", ha="left", fontsize=9.5)

    ax.plot([], [], color=CUOPT_C, linewidth=8, label="distributed reduces peak")
    ax.plot([], [], color="#1A1A1A", linewidth=1.2, label="parity (1×)")
    ax.plot([], [], color="#1A1A1A", linewidth=1.0, linestyle=(0, (3, 2)),
            label=f"geomean ({gm:.2f}×)")
    leg = ax.legend(loc="lower right", frameon=True, fontsize=12,
                    fancybox=False, edgecolor="#CCCCCC", framealpha=1.0)
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


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tol", choices=("1e-4", "1e-6"), default="1e-6")
    p.add_argument("--csv", default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_cli()
    csv_path = Path(args.csv or (HERE / f"results_endtoend_tol{args.tol}.csv"))
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    out_dir = Path(args.out_dir or (HERE / "plots" / args.tol))
    pairs = load_pairs(csv_path)
    plot_grouped(pairs, out_dir / "memory_peak.png", args.tol)
    plot_ratio(pairs, out_dir / "memory_ratio.png", args.tol)

    db = load_dist_vs_base(csv_path)
    plot_dist_vs_base(db, out_dir / "memory_dist_vs_base.png", args.tol)
    plot_dist_vs_base_ratio(db, out_dir / "memory_dist_vs_base_ratio.png",
                            args.tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
