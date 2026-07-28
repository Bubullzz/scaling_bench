#!/usr/bin/env python3
"""Named instance subsets used by plot / report scripts.

ADDENDUM = Mittelmann LPfeas ADDENDUM (Oliver Hinder large-scale set),
as listed on plato.asu.edu/ftp/lpfeas.html. Stems match what our
results CSVs use after _stem() (gurobi-presolved suffix stripped,
mcf files keep the long multicommodity-flow-instance_* name).
"""

from __future__ import annotations

# Official Mittelmann ADDENDUM names ↔ our CSV stems.
# Missing locally (not in results yet):
#   mcf_5000_100_400  (we bench mcf_5000_100_250 instead)
#   prod_100_300_02
ADDENDUM_STEMS: frozenset[str] = frozenset({
    "heat-source-easy",
    "multicommodity-flow-instance_2500_100_500",   # mcf_2500_100_500
    "multicommodity-flow-instance_5000_100_250",   # stand-in for mcf_5000_100_400
    "multicommodity-flow-instance_5000_50_500",    # mcf_5000_50_500
    "mediterranean-shipping",
    "production-imventory",                        # production-inventory
    "qap-tho-150",
    "qap-wil-100",
    "supply-chain",
    "design_match",                                # synthetic-design-match
    "tsp-gaia-10m",
})

# Paths as registered in bench.INSTANCES (for docs / regenerating lists).
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
