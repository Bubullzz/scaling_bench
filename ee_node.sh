#!/usr/bin/env bash
# ee_node.sh -- executed on the GPU node by `crun -b -i ee_node.sh`.
#
# Boots a docker container that mirrors the interactive dev environment
# (same cuopt image, same NFS-backed scratch mount, same peer scratch
# read-only mounts) and runs ee_in_container.sh inside it. That script
# in turn launches endtoend_bench.py which sweeps the (solver, instance,
# n_gpus, tol) grid via runs_endtoend/.claims/ coordination.
#
# Runs UNPRIVILEGED (no root inside docker) -- the container image accepts
# RUNAS_UID/RUNAS_USER env vars to drop privileges to the invoking user.
#
# CLI args to this script (if any) are forwarded verbatim to
# endtoend_bench.py (via ee_in_container.sh). Typical uses:
#     ./ee_node.sh                       # default: --tol 1e-4 1e-6, all solvers
#     ./ee_node.sh --tol 1e-6            # 1e-6 sweep only
#     ./ee_node.sh --solver dpdlp        # only the D-PDLP solver
#
# Env knobs (all optional):
#     IMAGE       -- override the docker image (default katrines-cuopt-julia-dev:2004-112)
#     NCCL_DEBUG  -- verbosity for NCCL layer (default WARN; try INFO if debugging)
#
# NB: we KEEP NVLS enabled (NCCL_NVLS_ENABLE=1) but silence the resulting
# WARN spam by lowering NCCL_DEBUG to VERSION.
#
# What was happening:
#   NCCL 2.30+ turns NVLink SHARP (NVLS) ON by default on B200. NVLS wants
#   VMM-mapped buffers so it calls cuMemGetAddressRange(buf) to check;
#   cuopt-distributed allocates via cudaMalloc which VMM doesn't track,
#   so cuMemGetAddressRange returns 1 ("invalid argument") and NCCL
#   emits a WARN at include/alloc.h:383 -- literally once per collective.
#   Correctness is fine (NCCL falls back to non-NVLS silently) but the
#   log gets spammed 40x/s.
#
# NCCL_DEBUG=VERSION -> print version once at init, skip WARN/INFO/TRACE.
#   So we still get proof NCCL loaded (`NCCL version 2.x`) and any real
#   error stops the run, but the alloc-check WARN vanishes.
#   Bump to WARN or INFO temporarily if you need to debug a stall.

set -euo pipefail

echo "### ee_node boot: host=$(hostname) date=$(date -u +%FT%TZ) pid=$$ args='$*'"

# --- one-line PS4 with a helpful timestamp so `bash -x` output is readable ---
export PS4='+ [ee_node $(date +%H:%M:%S)] '
set -x

# ---------------------------------------------------------------------------
# 1) Discover the NFS mount of the invoking user's scratch.
# ---------------------------------------------------------------------------
# The node sees /home/scratch.<user>_gpu as an autofs-mounted NFS share.
# We need its (server, export) pair so we can re-mount it INSIDE the docker
# container as a proper NFS volume (docker cannot see the host autofs).

default_path="/home/scratch.vmostovoi_gpu"
scratch_path="${SCRATCH_PATH:-$default_path}"
scratch_string="$(basename "$scratch_path")"        # scratch.vmostovoi_gpu
tmp="${scratch_string#scratch.}"                    # vmostovoi_gpu
user_string="${tmp%_gpu*}"                          # vmostovoi
user_uid="$(id -u "$user_string")"

# `df --output=source $path` -> "server:/export/path"
mount_source="$(df --output=source "$scratch_path" | tail -n +2 | head -n 1)"
host_name="${mount_source%%:*}"
vol_name="${mount_source#*:}"
host_ip="$(getent hosts "$host_name" | awk 'NR==1 {print $1}')"

# fail loudly if any of those turned up empty (otherwise docker mount would
# silently break at the container boundary and everything looks fine here).
for v in scratch_path scratch_string user_string user_uid vol_name host_name host_ip; do
    [[ -z "${!v}" ]] && { echo "ee_node: variable $v is empty -- aborting" >&2; exit 2; }
done

# ---------------------------------------------------------------------------
# 2) Docker config.
# ---------------------------------------------------------------------------
image="${IMAGE:-nvcr.io/nvidian/dt-compute/katrines-cuopt-julia-dev:2004-112}"
# VERSION = print `NCCL version X.Y.Z+cudaXX` once and be quiet after.
# Suppresses the NVLS/cuMemGetAddressRange WARN spam. Bump to WARN/INFO
# to debug a stall.
nccl_debug="${NCCL_DEBUG:-VERSION}"
# Keep NVLS enabled (default in NCCL 2.30+); the WARN spam it causes on
# cudaMalloc'd buffers is already silenced by NCCL_DEBUG=VERSION above.
nccl_nvls="${NCCL_NVLS_ENABLE:-1}"
python_bin="/home/scratch.vmostovoi_gpu/.conda/envs/cuopt_dev_133/bin/python"

# Forward any CLI args to endtoend_bench.py through ee_in_container.sh.
# printf %q so weird chars survive the shell layers.
bench_args_q=()
for a in "$@"; do bench_args_q+=("$(printf '%q' "$a")"); done

echo "=== ee_node ==="
echo "host       : $(hostname)"
echo "date       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "scratch    : $scratch_string ($host_ip) -> $scratch_path"
echo "image      : $image"
echo "python     : $python_bin"
echo "NCCL_DEBUG : $nccl_debug"
echo "NCCL_NVLS  : $nccl_nvls"
echo "ee args    : $*"
echo "================"

# Read-only peer mounts (dataset / cuopt source / etc.) -- only mount those
# that actually exist on this node so we don't fail on newer or renamed shares.
extra_mounts=()
for peer in \
    /home/scratch.bbozkaya_gpu \
    /home/scratch.nblin_gpu_1 \
    /home/scratch.yboucher_gpu_1 \
    /home/scratch.svc_compute_arch \
    /home/scratch.cmaes_sw \
    /home/cuOptProject ; do
    if [[ -d "$peer" ]]; then
        extra_mounts+=("-v" "$peer:$peer")
    fi
done

in_container_script="$scratch_path/scaling_bench/ee_in_container.sh"

# ---------------------------------------------------------------------------
# 3) Run docker. The container inherits GPUs, gets a proper NFS mount of the
# user's scratch, and executes ee_in_container.sh which chains into python.
# ---------------------------------------------------------------------------
exec docker run --rm --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NCCL_DEBUG="$nccl_debug" \
    -e NCCL_NVLS_ENABLE="$nccl_nvls" \
    -e RUNAS_UID="$user_uid" \
    -e RUNAS_USER="$user_string" \
    -e HOST_IP="$host_ip" \
    -e PYTHONUNBUFFERED=1 \
    --security-opt=seccomp=unconfined \
    --mount "type=volume,volume-opt=o=addr=$host_ip,volume-opt=device=:$vol_name,volume-opt=type=nfs,source=$scratch_string,target=$scratch_path" \
    "${extra_mounts[@]}" \
    "$image" \
    /bin/bash "$in_container_script" "$@"
