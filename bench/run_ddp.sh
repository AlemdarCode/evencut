#!/usr/bin/env bash
# Runs the multi-GPU part of test 4 on a Linux machine with several NVIDIA GPUs (for example a
# rented one) and writes bench/results/ddp_batches*.md.
#
# Usage:  bench/run_ddp.sh <lengths-dir> [number-of-gpus]
#         BASE=1 bench/run_ddp.sh ...      also runs a larger (~90M parameter) model
#
# <lengths-dir> holds tinystories_tr.txt and dolly15k_gpt2.txt, written by
#   python bench/gpu_batches.py --tokens ... --save-lengths <dir>/tinystories_tr.txt
#   python bench/gpu_batches.py --jsonl ... --save-lengths <dir>/dolly15k_gpt2.txt
# Needs PyTorch with CUDA. Evencut is used from this checkout; its pure-Python backend is fast
# enough for these plans, so nothing has to be compiled.
set -euo pipefail
DATA=${1:?usage: bench/run_ddp.sh <lengths-dir> [gpus]}
GPUS=${2:-$(nvidia-smi -L | wc -l)}
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -c "import torch, evencut; print('torch', torch.__version__, 'cuda', torch.version.cuda,
'gpus', torch.cuda.device_count(), 'evencut', evencut.__version__, evencut.backend)"

run() { torchrun --standalone --nproc_per_node="$GPUS" bench/ddp_batches.py "$@"; }

# 1. a three-step dry run catches setup problems before the measured runs
run --lengths "$DATA/tinystories_tr.txt" --offset 20480 --steps 3 --warmup 1 --repeats 1 \
    --name dry-run --out-dir /tmp/evencut-dry-run

# 2. the measured runs
run --lengths "$DATA/tinystories_tr.txt" --offset 20480 --name TinyStories-Turkish --out-name ddp_batches
run --lengths "$DATA/dolly15k_gpt2.txt" --seq-len 1024 --global-batch 128 --name Dolly-15k \
    --out-name ddp_batches_dolly
if [ "${BASE:-0}" = "1" ]; then
    run --lengths "$DATA/tinystories_tr.txt" --offset 20480 --model base --name TinyStories-Turkish \
        --out-name ddp_batches_base
    run --lengths "$DATA/dolly15k_gpt2.txt" --seq-len 1024 --global-batch 128 --model base --name Dolly-15k \
        --out-name ddp_batches_dolly_base
fi
echo "done: bench/results/ddp_batches*.md"
