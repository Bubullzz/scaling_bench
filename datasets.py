#!/usr/bin/env python3
"""Named instance subsets used by plot / report scripts.

Stems match what end-to-end CSVs use after stripping paths, compression
suffixes, and optional ``_gurobi_presolved`` (see plot_speedup_1v1._stem).

Sets intentionally overlap: ``bench.INSTANCES`` remains the unique
execution grid; these are filters for plotting / reporting only.

Missing vs official Mittelmann ADDENDUM table (not benched locally):
  mcf_5000_100_400  (we use multicommodity-flow-instance_5000_100_250)
  prod_100_300_02
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Overlapping stem sets (canonical names for --dataset)
# ---------------------------------------------------------------------------

ADDENDUM_STEMS: frozenset[str] = frozenset({
    "heat-source-easy",
    "multicommodity-flow-instance_2500_100_500",
    "multicommodity-flow-instance_5000_100_250",
    "multicommodity-flow-instance_5000_50_500",
    "mediterranean-shipping",
    # prod_100_300_02
    "production-imventory",
    "qap-tho-150",
    "qap-wil-100",
    "supply-chain",
    "design_match",
    "tsp-gaia-10m",
})

# Original PDLP / cuPDLPx Google-Hinder paper instances (+ hard variants).
PDLP_STEMS: frozenset[str] = frozenset({
    "design_match",
    "tsp-gaia-10m",
    # tsp gaia 100m
    "heat-source-easy",
    "heat-source-hard",
    "production-imventory",
    "qap-tho-150",
    "qap-wil-100",
    "world-shipping",
    "mediterranean-shipping",
    "supply-chain",
})

# D-PDLP huge-scale curated set (MCF + design-match + QAP + zib03).
DPDLP_STEMS: frozenset[str] = frozenset({
    "zib03",
    #pagerank 1
    #pagerank 2
    #pagerank 3
    "multicommodity-flow-instance_2500_100_500",
    "multicommodity-flow-instance_5000_100_250",
    "multicommodity-flow-instance_5000_50_500",
    "design_match",
    "wil50",
    "lipa50a",
    "lipa50b",
    "tai50a",
    "tai50b",
    #ds1
    #ds2
})

INDUSTRY_STEMS: frozenset[str] = frozenset({
    "psr_100",
    "C5_bigger_sanitized",
    "C5_baseline_sanitized",
    "ELMOD_876_10_noVEnames",
    "BEAM_4032_11_8_CLI",
    "amazon_lp003",
    "amazon_lp004",
    "model_de_20_scenarios",
    "model_de_50_scenarios",
    "model_de_100_scenarios",
})

OPEN_ENERGY_STEMS: frozenset[str] = frozenset({
    "zen-garden-eur-PI-28-200ts",
    "IESA-Opt-NL-10-1h",
    "IESA-Opt-NL-5-3h",
    "TIMES-STEM-15-1h",
    "zen-garden-eur-PI-constrained-expansion-28-100ts",
    "pypsa-de-elec-60-1h",
    "ethos_fine_europe_60tp-175-720ts",
    "pypsa-eur-elec-100-3h",
    "zen-garden-eur-PI-no-storage-28-100ts",
    "times-ireland-noco2-40-1ts",
    "pypsa-eur-sec-50-24h",
})

# Full Hans Mittelmann LPFeas set (local copy: scratch.cmaes_sw/hans_lps).
LPFEAS_STEMS: frozenset[str] = frozenset({
    "16_n14",
    "Dual2_5000",
    "L1_sixm1000obs",
    "L1_sixm250obs",
    "L2CTA3D",
    "Linf_520c",
    "Primal2_1000",
    "a2864",
    "bdry2",
    "cont1",
    "cont11",
    "datt256_lp",
    "degme",
    "dlr1",
    "dlr2",
    "ex10",
    "fhnw-binschedule1",
    "fome13",
    "graph40-40",
    "i_n13",
    "irish-electricity",
    "karted",
    "lo10",
    "long15",
    "neos",
    "neos-3025225",
    "neos-5052403-cygnet",
    "neos-5251015",
    "neos3",
    "netlarge1",
    "netlarge2",
    "netlarge3",
    "netlarge6",
    "ns1687037",
    "ns1688926",
    "nug08-3rd",
    "pds-100",
    "physiciansched3-3",
    "qap15",
    "rail02",
    "rail4284",
    "rmine15",
    "s100",
    "s250r10",
    "s82",
    "savsched1",
    "scpm1",
    "set-cover-model",
    "shs1023",
    "square15",
    "square41",
    "stat96v2",
    "stormG2_1000",
    "stp3d",
    "supportcase10",
    "supportcase19",
    "thk_48",
    "thk_63",
    "tpl-tub-ws1617",
    "wide15",
    "woodlands09",
})

# Overlapping named subsets that contribute to the aggregate "all".
OVERLAPPING_DATASET_STEMS: dict[str, frozenset[str]] = {
    "addendum":     ADDENDUM_STEMS,
    "pdlp":         PDLP_STEMS,
    "dpdlp":        DPDLP_STEMS,
    "industry":     INDUSTRY_STEMS,
    "open-energy":  OPEN_ENERGY_STEMS,
    "lpfeas":       LPFEAS_STEMS,
}

# Full local suite plotted as "all": union of every named overlapping set.
ALL_STEMS: frozenset[str] = frozenset().union(*OVERLAPPING_DATASET_STEMS.values())

DATASET_STEMS: dict[str, frozenset[str]] = {
    "all": ALL_STEMS,
    **OVERLAPPING_DATASET_STEMS,
}

# Stable CLI ordering for named (stored) datasets.
NAMED_DATASET_NAMES: tuple[str, ...] = tuple(DATASET_STEMS)

# Paths as registered in bench.INSTANCES (docs only).
ADDENDUM_PATHS: list[str] = [
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/heat-source-easy.mps",
    "/home/scratch.cmaes_sw/large-scale-LP-test-problems/large-problem-instances/multicommodity-flow-instance_2500_100_500.mps.gz",
    "/home/scratch.cmaes_sw/large-scale-LP-test-problems/large-problem-instances/multicommodity-flow-instance_5000_100_250.mps",
    "/home/scratch.cmaes_sw/large-scale-LP-test-problems/large-problem-instances/multicommodity-flow-instance_5000_50_500.mps",
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/mediterranean-shipping.mps",
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/production-imventory.mps",
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/qap-tho-150.mps",
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/qap-wil-100.mps",
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/supply-chain.mps",
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/design_match.mps",
    "/home/scratch.vmostovoi_gpu/datasets/big_lp/google/tsp-gaia-10m.mps",
]


def _normalize_grid_stem(stem: str) -> str:
    """Match plot_speedup_1v1._stem() normalization on grid path stems."""
    for suf in ("_PSLP_presolved", "_gurobi_presolved"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    if stem == "psr-100":
        stem = "psr_100"
    return stem


def grid_stems() -> frozenset[str]:
    """Stems in the end-to-end sweep grid (NO_PRESOLVE_THESE = raw + no-presolve)."""
    import endtoend_bench as ee
    import bench
    return frozenset(
        _normalize_grid_stem(bench._stem_of(p))
        for p in ee._all_instances()
    )


def validate_datasets(*, grid: frozenset[str] | None = None) -> None:
    """Raise ValueError if any registered stem is absent from the sweep grid."""
    grid = grid if grid is not None else grid_stems()
    errors: list[str] = []
    for name, stems in DATASET_STEMS.items():
        missing = sorted(stems - grid)
        if missing:
            errors.append(f"  {name}: {', '.join(missing)}")
    if errors:
        raise ValueError(
            "Dataset stem(s) not in end-to-end grid (typo or retired instance?):\n"
            + "\n".join(errors)
        )


def dataset_filter(dataset: str) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    """Return (allow, exclude) stem filters for plot scripts.

    ``non-addendum`` is the complement of ADDENDUM (not a stored set).
    ``all`` is the union of every named overlapping set (including lpfeas).
    """
    if dataset == "non-addendum":
        return None, ADDENDUM_STEMS
    if dataset not in DATASET_STEMS:
        raise KeyError(
            f"unknown dataset {dataset!r}; "
            f"choose from {list(DATASET_STEMS)} or 'non-addendum'"
        )
    return DATASET_STEMS[dataset], None


def plot_out_dir(base, tol: str, dataset: str):
    """Output directory for speedup plots under plots/<tol>/<dataset>/."""
    from pathlib import Path
    return Path(base) / tol / dataset
