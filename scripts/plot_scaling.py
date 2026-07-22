#!/usr/bin/env python3
"""
scaling_bench/scripts/plot_scaling.py

Generate strong-scaling (and optionally weak-scaling) plots from
results.csv. Uses matplotlib + pandas + numpy. We point at the
d_pdlp_env conda env because that's where matplotlib is installed on
this machine; you can also just run with any python3 that has
matplotlib + pandas + numpy.

Outputs (all based on step_time_s, i.e. setup excluded -- that's what
we care about for scaling, since setup grows with N):

  plots/strong/<stem>_speedup.png        - log-log speedup vs N, ideal y=x
  plots/strong/<stem>_time_per_step.png  - per-step time vs N
  plots/strong/<stem>_efficiency.png     - speedup/N vs N (linear)
  plots/strong/all_speedup.png           - all problems overlaid

  plots/weak/<group>_time_per_step.png   - flat under ideal weak scaling
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except ImportError as e:
    print(
        f"matplotlib/pandas/numpy not available ({e}); use the d_pdlp_env "
        "python: /home/scratch.vmostovoi_gpu/.conda/envs/d_pdlp_env/bin/python3",
        file=sys.stderr,
    )
    sys.exit(2)

try:
    import yaml  # PyYAML, optional (only needed for weak scaling)
except ImportError:
    yaml = None


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results.csv"
SUMMARY = ROOT / "results_summary.csv"
WEAK_YAML = ROOT / "weak_scaling_groups.yaml"
OUT = ROOT / "plots"


def _setup_axes_loglog(ax, ns, xlabel, ylabel, title):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(n)) for n in ns])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)


def plot_strong(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # PRIMARY metric for strong scaling: time spent inside the PDLP step loop.
    # This is total_time_s - setup_time_s, i.e. the part that scaling
    # actually targets. Using total_time_s would penalize larger N because
    # setup (METIS / NCCL bootstrap / scaling / sigma_max) grows with N.
    metric = "step_time_s"
    tpi    = "time_per_step_ms"

    # Drop reps that don't have a derivable step time.
    df = df.dropna(subset=[metric, "n_gpus_requested"])
    if df.empty:
        print(f"strong: no rows with {metric}; nothing to plot.", file=sys.stderr)
        return
    df = df[~df["is_warmup"].astype(bool)]

    # Median per (stem, n_gpus).
    by = (
        df.groupby(["stem", "n_gpus_requested"])
        .agg(
            t=(metric, "median"),
            tpi=(tpi, "median"),
        )
        .reset_index()
    )

    # Per-problem baseline = smallest N for that problem with data.
    baselines = (
        by.sort_values(["stem", "n_gpus_requested"])
        .groupby("stem")
        .first()
        .reset_index()
        .rename(columns={"n_gpus_requested": "baseline_n", "t": "t_baseline"})
        [["stem", "baseline_n", "t_baseline"]]
    )

    all_speedup_rows = []

    for stem in sorted(by["stem"].unique()):
        sub = by.loc[by["stem"] == stem].sort_values("n_gpus_requested")
        if len(sub) < 1:
            continue
        b_row = baselines.loc[baselines["stem"] == stem].iloc[0]
        b_t = float(b_row["t_baseline"])
        ns = sub["n_gpus_requested"].to_numpy(dtype=float)
        ts = sub["t"].to_numpy(dtype=float)
        tpis = sub["tpi"].to_numpy(dtype=float)
        speedup = b_t / ts
        ideal = ns / b_row["baseline_n"]
        efficiency = speedup / (ns / b_row["baseline_n"])

        for n, sp in zip(ns, speedup):
            all_speedup_rows.append({"stem": stem, "n_gpus": n, "speedup": sp})

        # speedup (based on step_time_s, NOT total wall time)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(ns, speedup, "o-", label="measured (step time only)", linewidth=2)
        ax.plot(ns, ideal, "--", label=f"ideal (T/{int(b_row['baseline_n'])} GPU)", alpha=0.6)
        _setup_axes_loglog(
            ax, ns,
            xlabel="GPUs (N)",
            ylabel=f"speedup vs N={int(b_row['baseline_n'])}",
            title=f"{stem}: strong scaling (step time, setup excluded)",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_speedup.png", dpi=140)
        plt.close(fig)

        # time per step (ms) - the rawest scaling diagnostic
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(ns, tpis, "o-", linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.set_xlabel("GPUs (N)")
        ax.set_ylabel("time / step (ms)")
        ax.set_title(f"{stem}: per-step time vs N (setup excluded)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_time_per_step.png", dpi=140)
        plt.close(fig)

        # efficiency
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(ns, efficiency, "o-", linewidth=2)
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6, label="ideal")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.set_xlabel("GPUs (N)")
        ax.set_ylabel("efficiency (speedup / N)")
        ax.set_title(f"{stem}: parallel efficiency")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_efficiency.png", dpi=140)
        plt.close(fig)

    # All problems overlaid.
    if all_speedup_rows:
        ov = pd.DataFrame(all_speedup_rows)
        ns_sorted = np.array(sorted(ov["n_gpus"].unique()))
        fig, ax = plt.subplots(figsize=(7, 6))
        for stem in sorted(ov["stem"].unique()):
            s = ov.loc[ov["stem"] == stem].sort_values("n_gpus")
            ax.plot(s["n_gpus"].to_numpy(dtype=float), s["speedup"].to_numpy(dtype=float), "o-", label=stem)
        ax.plot(ns_sorted, ns_sorted, "--", color="gray", alpha=0.6, label="ideal y=x")
        _setup_axes_loglog(
            ax, ns_sorted,
            xlabel="GPUs (N)",
            ylabel="speedup (relative to each problem's smallest N)",
            title="strong scaling overlay (step time)",
        )
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(out_dir / "all_speedup.png", dpi=140)
        plt.close(fig)


def plot_weak(df: pd.DataFrame, groups: list[dict], out_dir: Path) -> None:
    if not groups:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = "time_per_step_ms"  # setup-excluded; ideal weak scaling is flat
    total = "total_time_s"
    df = df[~df["is_warmup"].astype(bool)]
    df = df.dropna(subset=[metric, "n_gpus_requested"])

    for g in groups:
        name = g.get("name", "weak_group")
        pairs = g.get("pairs", [])
        if not pairs:
            continue
        rows = []
        for pair in pairs:
            stem = pair.get("stem")
            n = pair.get("n_gpus")
            if stem is None or n is None:
                continue
            mask = (df["stem"] == stem) & (df["n_gpus_requested"] == n)
            if mask.sum() == 0:
                print(f"weak [{name}]: no data for stem={stem} N={n}", file=sys.stderr)
                continue
            sub = df.loc[mask]
            rows.append(
                dict(
                    stem=stem,
                    n_gpus=n,
                    time_per_step_ms=float(sub[metric].median()),
                    total_time_s=float(sub[total].median()),
                    nnz=int(sub["nnz"].iloc[0]) if pd.notna(sub["nnz"].iloc[0]) else None,
                )
            )
        if not rows:
            continue

        gdf = pd.DataFrame(rows).sort_values("n_gpus")
        ns = gdf["n_gpus"].to_numpy()
        tpss = gdf["time_per_step_ms"].to_numpy()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(ns, tpss, "o-", linewidth=2, label=name)
        ax.axhline(tpss[0], linestyle="--", alpha=0.5,
                   label=f"ideal (= {tpss[0]:.2f} ms)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.set_xlabel("GPUs (N), problem grows with N")
        ax.set_ylabel("time / step (ms)")
        ax.set_title(f"weak scaling: {name} (step time only)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        # Annotate each point with the stem so the user can verify the
        # problem family is what they think it is.
        for _, r in gdf.iterrows():
            ax.annotate(
                r["stem"],
                xy=(r["n_gpus"], r["time_per_step_ms"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize="x-small",
            )
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_time_per_step.png", dpi=140)
        plt.close(fig)


def main() -> int:
    if not RESULTS.exists():
        print(f"missing {RESULTS}; run aggregate.py first", file=sys.stderr)
        return 1
    df = pd.read_csv(RESULTS)
    plot_strong(df, OUT / "strong")

    # Weak scaling is opt-in: needs weak_scaling_groups.yaml.
    if WEAK_YAML.exists():
        if yaml is None:
            print(
                f"{WEAK_YAML} exists but PyYAML is missing; skipping weak plots",
                file=sys.stderr,
            )
        else:
            with WEAK_YAML.open("r") as f:
                spec = yaml.safe_load(f) or {}
            plot_weak(df, spec.get("groups", []), OUT / "weak")

    print(f"Plots written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
