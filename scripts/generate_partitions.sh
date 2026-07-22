#!/usr/bin/env bash
#
# scaling_bench/scripts/generate_partitions.sh
#
# OPTIONAL pre-step. Distributed PDLP partitions the bipartite constraint
# graph with METIS. METIS is not deterministic across runs unless its
# RNG is seeded or partitions are pre-built and reused. To pin the
# partition (so rep-to-rep variance in step time reflects compute
# noise only, not partitioning noise), this script generates one
# partition file per (problem, n_gpus) pair into partitions/.
#
# It uses the metis_tests binary if you have one built locally; the
# default path below is a sensible guess. If your binary lives
# elsewhere, set METIS_TESTS to the path before invoking.
#
# Usage:
#   bash scripts/generate_partitions.sh
#       [--problems FILE] [--gpu-counts FILE] [--out DIR] [--force]
#
# If you skip this step, METIS will partition in-process during each
# cuopt_cli run; aggregate.py will report a higher
# time_per_step_ms_cov, which is your signal to come back here.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$HERE/.." >/dev/null 2>&1 && pwd)"

PROBLEMS_FILE="$ROOT/problems.list"
GPUS_FILE="$ROOT/gpu_counts.list"
OUT_DIR="$ROOT/partitions"
FORCE=0
METIS_TESTS="${METIS_TESTS:-/home/scratch.vmostovoi_gpu/metis_tests/build/metis_tests}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --problems)   PROBLEMS_FILE="$2"; shift 2 ;;
    --gpu-counts) GPUS_FILE="$2"; shift 2 ;;
    --out)        OUT_DIR="$2"; shift 2 ;;
    --force)      FORCE=1; shift ;;
    --binary)     METIS_TESTS="$2"; shift 2 ;;
    -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$METIS_TESTS" ]]; then
  cat >&2 <<EOF
generate_partitions.sh: metis_tests binary not found at:
  $METIS_TESTS

This step is optional. Without pre-built partitions, cuopt_cli will
run METIS in-process for each rep, which adds setup variance but does
not affect correctness. Re-run with METIS_TESTS=/path/to/metis_tests
if you want to cache.
EOF
  exit 0
fi

mkdir -p "$OUT_DIR"

read_list() { sed -e 's/#.*//' -e 's/[[:space:]]\+$//' "$1" | grep -vE '^[[:space:]]*$'; }

stem_of() {
  local p="$1"
  p="$(basename "$p")"
  p="${p%.gz}"; p="${p%.bz2}"; p="${p%.lz4}"
  p="${p%.mps}"; p="${p%.lp}"; p="${p%.qps}"
  echo "$p"
}

mapfile -t PROBLEMS < <(read_list "$PROBLEMS_FILE")
mapfile -t GPUS     < <(read_list "$GPUS_FILE")

for problem in "${PROBLEMS[@]}"; do
  [[ -r "$problem" ]] || { echo "skip unreadable: $problem"; continue; }
  stem="$(stem_of "$problem")"
  for n in "${GPUS[@]}"; do
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    [[ "$n" -ge 1 ]]       || continue
    out="$OUT_DIR/${stem}__N${n}.part"
    if [[ -s "$out" ]] && [[ "$FORCE" == 0 ]]; then
      echo "skip existing: $out"
      continue
    fi
    echo "--- $stem  N=$n ---"
    # NB: metis_tests CLI shape is project-specific. The expected
    # invocation is something like:
    #   metis_tests --mps <mps> --nparts <N> --out <part>
    # Adjust this line to match your local metis_tests if different.
    "$METIS_TESTS" --mps "$problem" --nparts "$n" --out "$out" \
      || { echo "  WARN: metis_tests failed for $stem N=$n"; continue; }
  done
done

echo
echo "Done. Generated partitions in $OUT_DIR."
echo "To actually have cuopt_cli USE these, add 'multi_gpu_partition_file = <path>'"
echo "to the params file per-run (or extend run_sweep.sh to pass it through)."
