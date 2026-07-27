#!/usr/bin/env bash
# ee_submit.sh -- submit ONE end-to-end benchmark worker to slurm/crun.
#
# Uses `crun -b` (batch, detached: writes slurm-<jobid>.out and returns
# immediately). The allocated node then runs ee_node.sh, which in turn
# runs docker + ee_in_container.sh + endtoend_bench.py.
#
# Any CLI args to this script are forwarded to endtoend_bench.py, e.g.
#     ./ee_submit.sh                       # tol=[1e-4, 1e-6], all solvers
#     ./ee_submit.sh --tol 1e-6            # 1e-6 only
#     ./ee_submit.sh --solver dpdlp        # dpdlp only
#
# To launch many workers in a loop see ee_loop.sh.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_SCRIPT="$HERE/ee_node.sh"

# GPU + wall-time knobs (safe defaults for our biggest instances; a full
# 15-instance sweep at tol=1e-6 needs ~4h at worst-case time_limit=3600s).
GPU_COUNT="${EE_GPU_COUNT:-8}"
WALL_TIME="${EE_WALL_TIME:-4:00:00}"
GPU_QUERY="${EE_GPU_QUERY:-gpu.product_name=*B200* and cpu.arch=x86_64}"

# Comma-separated slurm hostname blacklist forwarded to sbatch via
# `--exclude`. Keeps sick nodes (bad NCCL/GPU driver state, flaky NFS
# cache, hung containers) out of the pool without waiting for the ops
# team to reboot them. The default entry `umbriel-b200-042` was added
# on 2026-07-23 after that node re-crashed 16 cuopt-distributed runs
# in a row with `ncclCommInitAll failed` across two consecutive slurm
# allocations -- see the "double-check post-claim" note in
# bench.run_one for full backstory.
#
# Override / extend by setting EE_EXCLUDE_NODES in the environment,
# e.g. `EE_EXCLUDE_NODES="umbriel-b200-042,umbriel-b200-021" ./ee_submit.sh`.
# Set to an empty string to disable exclusion.
EE_EXCLUDE_NODES="${EE_EXCLUDE_NODES-umbriel-b200-042}"

echo "### ee_submit: $(date -u +%FT%TZ) submitting one worker (gpus=$GPU_COUNT wall=$WALL_TIME)"
echo "### ee_submit: forwarding args: $*"
if [[ -n "$EE_EXCLUDE_NODES" ]]; then
    echo "### ee_submit: blacklist  : sbatch --exclude=$EE_EXCLUDE_NODES"
fi

# crun -b <script>  = batch mode. Per `crun --help`:
#   "Batch jobs are queued for execution if a node is not available
#    immediately."
# so the submission always succeeds and slurm queues the job when every
# matching node is busy. Do NOT combine with -i (which switches crun to
# interactive-with-command mode and refuses to queue -- you get
# "Error: Could not find any free nodes with available GPUs" instead).
# -ex   = exclusive node reservation (all 8 GPUs, no sharing).
# -g N  = allocate N GPUs.
# -q    = node query (product name + cpu arch).
# -t    = wall-clock cap.
# -sa   = pass raw args to slurm's sbatch. We use it to inject
#         `--exclude=<hostnames>` so a known-broken node is skipped
#         entirely -- see EE_EXCLUDE_NODES above.
# The script (ee_node.sh) is the positional trailing arg; extra CLI args
# to ee_submit.sh are forwarded and slurm's sbatch will pass them to the
# batch script as $1, $2, ..., where ee_node.sh re-forwards them to
# ee_in_container.sh and eventually to endtoend_bench.py.
CRUN_CMD=(crun -b -ex
    -t "$WALL_TIME"
    -g "$GPU_COUNT"
    -q "$GPU_QUERY")
if [[ -n "$EE_EXCLUDE_NODES" ]]; then
    CRUN_CMD+=(-sa "--exclude=$EE_EXCLUDE_NODES")
fi
CRUN_CMD+=("$NODE_SCRIPT" "$@")

exec "${CRUN_CMD[@]}"
