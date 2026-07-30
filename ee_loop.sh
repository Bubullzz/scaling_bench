#!/usr/bin/env bash
# ee_loop.sh -- submit MANY end-to-end workers in a loop.
#
# Each worker is a fully independent slurm job (via ee_submit.sh -> crun -b)
# that lands on its own node and runs endtoend_bench.py. Workers coordinate
# through runs_endtoend/.claims/ (atomic claim files), so it's safe to run
# as many as you like in parallel -- they'll partition the (solver, instance,
# n_gpus, tol) tuples between them and stale claims get reassigned after
# a few minutes so a dead worker doesn't strand work.
#
# Usage:
#     ./ee_loop.sh [N] [-- <args passed to endtoend_bench.py>]
#
# Examples:
#     ./ee_loop.sh 8                        # 8 workers, default tolerances
#     ./ee_loop.sh 12 -- --tol 1e-6         # 12 workers, only tol=1e-6
#     ./ee_loop.sh 4  -- --solver dpdlp     # 4 workers, D-PDLP only
#
#     # cuopt-base on a single shared B200 (does NOT take a full 8-GPU node):
#     EE_PROFILE=base ./ee_loop.sh 6 -- --solver cuopt-base
#     EE_PROFILE=base ./ee_loop.sh 4 -- --solver cuopt-base --tol 1e-4
#
# Env knobs (all optional; forwarded to ee_submit.sh):
#     EE_PROFILE      -- dist (default, 8×B200 exclusive) | base (1×B200 shared)
#     EE_SLEEP        -- seconds between submissions (default 5)
#     EE_GPU_COUNT    -- GPUs per worker
#     EE_GPU_QUERY    -- crun -q node query
#     EE_WALL_TIME    -- crun -t value  (default 4:00:00)
#     EE_EXCLUSIVE    -- 1/0 for crun -ex
#
# Tips:
#   - If the sweep has ~15 instances and each solver run has time_limit
#     3600s, a single worker takes up to ~15h to walk the whole grid
#     -- but with N workers in parallel this drops to ~15h/N of wall time.
#   - Don't submit far more workers than there are unclaimed tuples;
#     endtoend_bench.py exits cleanly if it finds no work.
#   - Keep 8-GPU solvers (cuopt-distributed / dpdlp) on EE_PROFILE=dist
#     and cuopt-base on EE_PROFILE=base so scarce full-node B200 slots
#     aren't wasted on N=1 runs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N="${1:-4}"
shift || true
# Drop an optional literal "--" separator so `./ee_loop.sh 8 -- --tol 1e-6` is
# equivalent to `./ee_loop.sh 8 --tol 1e-6`.
if [[ "${1:-}" == "--" ]]; then shift; fi
FORWARD=("$@")

sleep_s="${EE_SLEEP:-5}"

if ! [[ "$N" =~ ^[0-9]+$ ]] || (( N < 1 )); then
    echo "ee_loop: N must be a positive integer, got '$N'" >&2
    echo "usage: $0 [N] [-- <endtoend_bench.py args...>]" >&2
    exit 2
fi

echo "### ee_loop: $(date -u +%FT%TZ) submitting $N worker(s)  profile=${EE_PROFILE:-dist}"
echo "### ee_loop: forwarding to endtoend_bench.py: ${FORWARD[*]:-<none>}"

for (( i=1; i<=N; i++ )); do
    echo "### ee_loop [$i/$N] submitting..."
    # ee_submit.sh -> crun -b returns immediately after slurm accepts the
    # job. If crun cannot find a matching free node right now it exits
    # non-zero -- we let that surface so the caller can see it (no retry).
    "$HERE/ee_submit.sh" "${FORWARD[@]}"
    if (( i < N )); then
        sleep "$sleep_s"
    fi
done

echo "### ee_loop: all $N submissions done. Watch progress with:"
echo "    ls -lt /home/scratch.vmostovoi_gpu/scaling_bench/slurm-*.out | head"
echo "    ls /home/scratch.vmostovoi_gpu/scaling_bench/runs_endtoend/*/*/  # per-run logs"
echo "    /home/scratch.vmostovoi_gpu/.conda/envs/cuopt_dev_133/bin/python endtoend_build_csv.py --tol 1e-4"
