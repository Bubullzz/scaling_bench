#!/usr/bin/env bash
#
# scaling_bench/scripts/run_sweep.sh
#
# Drive the strong/weak scaling sweep of Distributed PDLP. For every
# (problem, n_gpus) pair listed in problems.list / gpu_counts.list,
# invoke cuopt_cli with the fixed parameters in
# config/distributed_pdlp.params (5000 iterations, all six tolerances
# pinned to 1e-30 so the solver never short-circuits), do --warmups N
# warmup runs and --reps M timed reps, pin GPU selection via
# CUDA_VISIBLE_DEVICES=0,1,...,N-1 to keep the device set stable across
# reps, and write one log per run into runs/.
#
# Each log starts with a `### header` block that aggregate.py / parse_log.py
# read (problem path, stem, n_gpus_requested, rep id, started/finished
# timestamps, optional partition file, CUDA_VISIBLE_DEVICES). The rest
# is the cuopt_cli stdout/stderr.
#
# Usage:
#   bash scripts/run_sweep.sh [--reps 3] [--warmups 1] [--force] [--dry-run]
#       [--problems FILE] [--gpu-counts FILE] [--params FILE]
#       [--binary PATH] [--runs-dir DIR]
#
# Resume semantics: existing log files with non-zero size are skipped
# unless --force is passed. Warm-ups are always re-run (cheap to redo).

set -euo pipefail

# ----------------------------- defaults --------------------------------------
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$HERE/.." >/dev/null 2>&1 && pwd)"

REPS=3
WARMUPS=1
FORCE=0
DRY_RUN=0
PROBLEMS_FILE="$ROOT/problems.list"
GPUS_FILE="$ROOT/gpu_counts.list"
PARAMS_FILE="$ROOT/config/distributed_pdlp.params"
BINARY="/home/scratch.vmostovoi_gpu/bullshit_mostovoi_cuopt/cpp/build/cuopt_cli"
RUNS_DIR="$ROOT/runs"
PARTITIONS_DIR="$ROOT/partitions"

# ----------------------------- arg parse -------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps)         REPS="$2"; shift 2 ;;
    --warmups)      WARMUPS="$2"; shift 2 ;;
    --force)        FORCE=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --problems)     PROBLEMS_FILE="$2"; shift 2 ;;
    --gpu-counts)   GPUS_FILE="$2"; shift 2 ;;
    --params)       PARAMS_FILE="$2"; shift 2 ;;
    --binary)       BINARY="$2"; shift 2 ;;
    --runs-dir)     RUNS_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUNS_DIR" "$PARTITIONS_DIR"

# ----------------------------- input validation ------------------------------
[[ -r "$PROBLEMS_FILE" ]] || { echo "missing $PROBLEMS_FILE" >&2; exit 2; }
[[ -r "$GPUS_FILE"     ]] || { echo "missing $GPUS_FILE" >&2; exit 2; }
[[ -r "$PARAMS_FILE"   ]] || { echo "missing $PARAMS_FILE" >&2; exit 2; }
if [[ "$DRY_RUN" == 0 ]] && [[ ! -x "$BINARY" ]]; then
  echo "missing or non-executable binary: $BINARY" >&2
  echo "(pass --binary <path> to override, or --dry-run to skip execution)" >&2
  exit 2
fi

# Visible GPU count: prefer nvidia-smi if present, else parse
# CUDA_VISIBLE_DEVICES (default to 8 if unset, the user can override
# via gpu_counts.list).
visible_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -cE '^[0-9]+$'
  else
    echo 0
  fi
}
HAVE_GPUS=$(visible_gpus)
if [[ "$DRY_RUN" == 0 ]] && [[ "$HAVE_GPUS" -eq 0 ]]; then
  echo "warning: could not detect any GPUs; sweep will likely fail" >&2
fi

# ----------------------------- helpers ---------------------------------------
read_list() {
  # strip comments + blank lines from a list file
  sed -e 's/#.*//' -e 's/[[:space:]]\+$//' "$1" | grep -vE '^[[:space:]]*$'
}

iso_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

stem_of() {
  local p="$1"
  p="$(basename "$p")"
  p="${p%.gz}"
  p="${p%.bz2}"
  p="${p%.lz4}"
  p="${p%.mps}"
  p="${p%.lp}"
  p="${p%.qps}"
  echo "$p"
}

# Run one (problem, n_gpus, rep) combination. Args:
#   $1: problem path (abs)
#   $2: stem
#   $3: n_gpus
#   $4: rep id ("1", "2", ... or "W0", "W1", ... for warmups)
run_one() {
  local problem="$1" stem="$2" n="$3" rep="$4"
  local logf="$RUNS_DIR/${stem}__N${n}__rep${rep}.log"

  if [[ -s "$logf" ]] && [[ "$FORCE" == 0 ]] && [[ "$rep" != W* ]]; then
    echo "  skip (resume): $(basename "$logf")"
    return 0
  fi

  local part_file=""
  # If a pre-built partition exists under partitions/<stem>__N<n>.part,
  # surface it in the header (the params file does NOT reference it; if
  # you want it actually used, add multi_gpu_partition_file=... to the
  # params, or do that here at execution time).
  if [[ -s "$PARTITIONS_DIR/${stem}__N${n}.part" ]]; then
    part_file="$PARTITIONS_DIR/${stem}__N${n}.part"
  fi

  local started; started="$(iso_now)"
  local devices=""
  if [[ "$n" -gt 0 ]]; then
    devices="$(seq -s, 0 $((n - 1)))"
  fi

  {
    echo "### scaling_bench run header"
    echo "### problem=$problem"
    echo "### stem=$stem"
    echo "### n_gpus_requested=$n"
    echo "### rep=$rep"
    echo "### cuda_visible_devices=$devices"
    echo "### partition_file=${part_file:-NONE}"
    echo "### started=$started"
    echo "### params_file=$PARAMS_FILE"
    echo "### binary=$BINARY"
    echo "### --- begin cuopt_cli output ---"
  } > "$logf"

  if [[ "$DRY_RUN" == 1 ]]; then
    echo "  (dry-run) would: CUDA_VISIBLE_DEVICES=$devices $BINARY --params-file $PARAMS_FILE $problem"
    echo "### dry_run=1" >> "$logf"
    echo "### finished=$(iso_now)" >> "$logf"
    return 0
  fi

  # Run. We override distributed_pdlp_num_gpus on the command line so the
  # params file can stay fixed; cuopt_cli reads params first then CLI
  # flags override (per its argparse precedence). Actually
  # distributed_pdlp_num_gpus isn't exposed as a top-level CLI arg in
  # this build, so instead we restrict the device set via
  # CUDA_VISIBLE_DEVICES and let the binary auto-detect.
  set +e
  CUDA_VISIBLE_DEVICES="$devices" \
    "$BINARY" --params-file "$PARAMS_FILE" "$problem" >> "$logf" 2>&1
  local ec=$?
  set -e

  {
    echo "### exit_code=$ec"
    echo "### finished=$(iso_now)"
  } >> "$logf"

  # Classify success/failure cheaply (without invoking the Python
  # parser, which would be overkill per run). We treat any non-zero
  # exit code, or a missing "Status: ..." line, as a failure.
  local fail_reason=""
  if [[ "$ec" -ne 0 ]]; then
    fail_reason="exit=$ec"
  elif ! grep -qE '^Status: ' "$logf"; then
    fail_reason="no Status line"
  fi
  if [[ -n "$fail_reason" ]]; then
    # Try to capture the most informative line in the log as the
    # one-line failure summary. Order matches parse_log.py's
    # _FAILURE_PATTERNS.
    local excerpt
    excerpt=$(grep -E -m1 \
      -e 'A numerical error was encountered' \
      -e 'out of memory|std::bad_alloc|cudaError(MemoryAllocation|OutOfMemory)|CUDA_ERROR_OUT_OF_MEMORY' \
      -e 'NCCL error|ncclResult|ncclInternal|ncclSystem' \
      -e 'failed to parse|MPS reader|invalid MPS' \
      -e 'validation error' \
      -e 'cudaErrorOperatingSystem|cudaError|CUDA_ERROR' \
      -e 'terminate called|what\(\):|Segmentation fault|Aborted' \
      "$logf" 2>/dev/null | head -c 200)
    if [[ -n "$excerpt" ]]; then
      fail_reason="$fail_reason :: $excerpt"
    fi
    FAIL_LOG+=("$stem  N=$n  rep=$rep  [$fail_reason]")
    echo "  WARN: $fail_reason  ($(basename "$logf"))"
  fi
}

# Accumulator for failed runs; printed as a single summary at the end.
declare -a FAIL_LOG=()

# ----------------------------- main loop -------------------------------------
mapfile -t PROBLEMS < <(read_list "$PROBLEMS_FILE")
mapfile -t GPUS     < <(read_list "$GPUS_FILE")

if [[ "${#PROBLEMS[@]}" -eq 0 ]]; then
  echo "no problems in $PROBLEMS_FILE; add some MPS paths and retry." >&2
  exit 2
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "no GPU counts in $GPUS_FILE; add 1 2 4 8 ... and retry." >&2
  exit 2
fi

echo "=== scaling_bench sweep ==="
echo "  binary    : $BINARY"
echo "  params    : $PARAMS_FILE"
echo "  problems  : ${#PROBLEMS[@]} from $PROBLEMS_FILE"
echo "  gpu counts: ${GPUS[*]}  (have $HAVE_GPUS visible)"
echo "  warmups   : $WARMUPS    (rep ids W0, W1, ...)"
echo "  reps      : $REPS       (rep ids 1, 2, ...)"
echo "  runs dir  : $RUNS_DIR"
[[ "$DRY_RUN" == 1 ]] && echo "  *** DRY RUN ***"

total_runs=$(( ${#PROBLEMS[@]} * ${#GPUS[@]} * (WARMUPS + REPS) ))
echo "  total runs: $total_runs"
echo

idx=0
for problem in "${PROBLEMS[@]}"; do
  stem="$(stem_of "$problem")"
  if [[ "$DRY_RUN" == 0 ]] && [[ ! -r "$problem" ]]; then
    echo "skip unreadable problem: $problem"
    continue
  fi
  for n in "${GPUS[@]}"; do
    if [[ "$DRY_RUN" == 0 ]] && [[ "$n" -gt "$HAVE_GPUS" ]]; then
      echo "skip N=$n > available GPUs ($HAVE_GPUS)"
      continue
    fi

    echo "--- $stem  N=$n ---"
    for w in $(seq 0 $((WARMUPS - 1))); do
      idx=$((idx + 1))
      printf "  [%d/%d] warmup W%d ... " "$idx" "$total_runs" "$w"
      run_one "$problem" "$stem" "$n" "W$w" && echo "done"
    done
    for r in $(seq 1 "$REPS"); do
      idx=$((idx + 1))
      printf "  [%d/%d] rep %d ...    " "$idx" "$total_runs" "$r"
      run_one "$problem" "$stem" "$n" "$r" && echo "done"
    done
  done
done

echo
if [[ "${#FAIL_LOG[@]}" -gt 0 ]]; then
  echo "============================================================"
  echo "FAILED RUNS (${#FAIL_LOG[@]} of $total_runs):"
  echo "============================================================"
  for line in "${FAIL_LOG[@]}"; do
    echo "  $line"
  done
  echo
  echo "Hints:"
  echo "  - OOM at N=1 is expected for problems that don't fit on a"
  echo "    single GPU; the same problem will likely succeed at higher N."
  echo "  - For full classification + per-mode counts, run:"
  echo "      python3 scripts/aggregate.py"
  echo "    and look at the failure_modes column in results_summary.csv"
  echo "    or the failed-runs block printed at the end."
  echo
fi
echo "Sweep complete. Next: python3 scripts/aggregate.py"
