#!/usr/bin/env bash
# ee_in_container.sh -- entrypoint executed INSIDE the docker container.
#
# Called by ee_node.sh via `docker run ... /bin/bash ee_in_container.sh`.
# All this does:
#   1. print a small header (timestamp / hostname / uid / GPU list),
#   2. unbuffer python stdout,
#   3. cd into the shared scratch scaling_bench/ directory,
#   4. exec the endtoend_bench sweep with the conda-env python interpreter.
#
# endtoend_bench.py picks up unclaimed (solver, instance, n_gpus, tol) tuples
# from runs_endtoend/.claims/ and writes per-run logs; multiple containers on
# multiple nodes coordinate automatically. See bench.py docstring for details.
#
# CLI args passed to this script are forwarded to endtoend_bench.py, so you
# can add e.g. `--tol 1e-6` from ee_submit.sh without editing this file.

set -euo pipefail

echo "### ee_in_container: $(date -u +%FT%TZ) host=$(hostname) whoami=$(whoami) uid=$(id -u)"
echo "### ee_in_container args: $*"

set -x
export PYTHONUNBUFFERED=1

echo "### ee_in_container: nvidia-smi"
nvidia-smi -L

cd /home/scratch.vmostovoi_gpu/scaling_bench

echo "### ee_in_container: launching endtoend_bench.py $*"
exec /home/scratch.vmostovoi_gpu/.conda/envs/cuopt_dev_133/bin/python -u endtoend_bench.py "$@"
