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
# Profiles (set EE_PROFILE, or use the env knobs below directly):
#     EE_PROFILE=dist   (default)  8×B200 exclusive -- cuopt-distributed / D-PDLP
#     EE_PROFILE=base              1×B200, shared   -- cuopt-base only
#                                  (lands on a free B200 without taking the
#                                   whole 8-GPU node). Pair with: --solver cuopt-base
#
# Env knobs (all optional; override the profile):
#     EE_GPU_COUNT    -- GPUs to allocate      (dist: 8,  base: 1)
#     EE_GPU_QUERY    -- crun -q node query    (both: B200 x86_64)
#     EE_WALL_TIME    -- crun -t value         (default 4:00:00)
#     EE_EXCLUSIVE    -- 1 => crun -ex         (dist: 1,   base: 0)
#     EE_EXCLUDE_NODES -- comma-separated sbatch --exclude list
#
# Examples:
#     ./ee_submit.sh --tol 1e-6 --solver cuopt-distributed
#     EE_PROFILE=base ./ee_submit.sh --tol 1e-4 --solver cuopt-base
#     EE_GPU_COUNT=1 EE_EXCLUSIVE=0 \
#         ./ee_submit.sh --solver cuopt-base
#
# To launch many workers in a loop see ee_loop.sh.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_SCRIPT="$HERE/ee_node.sh"

PROFILE="${EE_PROFILE:-dist}"

case "$PROFILE" in
    dist|distributed|8gpu|b200)
        DEFAULT_GPU_COUNT=8
        DEFAULT_GPU_QUERY='gpu.product_name=*B200* and cpu.arch=x86_64'
        DEFAULT_EXCLUSIVE=1
        ;;
    base|1gpu|single)
        DEFAULT_GPU_COUNT=1
        # One B200 on a shared node (no -ex) — doesn't burn a full 8-GPU
        # allocation. Override EE_GPU_QUERY if you want a different SKU.
        DEFAULT_GPU_QUERY='gpu.product_name=*B200* and cpu.arch=x86_64'
        DEFAULT_EXCLUSIVE=0
        ;;
    *)
        echo "ee_submit: unknown EE_PROFILE='$PROFILE' (use dist|base)" >&2
        exit 2
        ;;
esac

GPU_COUNT="${EE_GPU_COUNT:-$DEFAULT_GPU_COUNT}"
WALL_TIME="${EE_WALL_TIME:-4:00:00}"
GPU_QUERY="${EE_GPU_QUERY:-$DEFAULT_GPU_QUERY}"
# Exclusive only makes sense when we want the whole node (8-GPU NCCL runs).
# For 1-GPU cuopt-base we share the node so we don't burn 7 idle GPUs.
EXCLUSIVE="${EE_EXCLUSIVE:-$DEFAULT_EXCLUSIVE}"

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
# Apply the B200 blacklist for both profiles (base also targets B200).
# Set EE_EXCLUDE_NODES="" to disable.
EE_EXCLUDE_NODES="${EE_EXCLUDE_NODES-umbriel-b200-042}"

echo "### ee_submit: $(date -u +%FT%TZ) profile=$PROFILE gpus=$GPU_COUNT exclusive=$EXCLUSIVE wall=$WALL_TIME"
echo "### ee_submit: query     : $GPU_QUERY"
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
# -ex   = exclusive node reservation (all GPUs, no sharing) -- used for
#         8-GPU NCCL runs; OFF for 1-GPU cuopt-base so we can land on a
#         shared / single-GPU node.
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
CRUN_CMD=(crun -b
    -t "$WALL_TIME"
    -g "$GPU_COUNT"
    -q "$GPU_QUERY")
if [[ "$EXCLUSIVE" == "1" ]]; then
    CRUN_CMD+=(-ex)
fi
if [[ -n "$EE_EXCLUDE_NODES" ]]; then
    CRUN_CMD+=(-sa "--exclude=$EE_EXCLUDE_NODES")
fi
CRUN_CMD+=("$NODE_SCRIPT" "$@")

exec "${CRUN_CMD[@]}"
