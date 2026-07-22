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
# Env knobs (all optional):
#     EE_SLEEP        -- seconds between submissions (default 5)
#     EE_GPU_COUNT    -- GPUs per worker (default 8)
#     EE_WALL_TIME    -- crun -t value  (default 4:00:00)
#
# Tips:
#   - If the sweep has ~15 instances and each solver run has time_limit
#     3600s, a single worker takes up to ~15h to walk the whole grid
#     -- but with N workers in parallel this drops to ~15h/N of wall time.
#   - Don't submit far more workers than there are unclaimed tuples;
#     endtoend_bench.py exits cleanly if it finds no work.

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

echo "### ee_loop: $(date -u +%FT%TZ) submitting $N worker(s)"
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
