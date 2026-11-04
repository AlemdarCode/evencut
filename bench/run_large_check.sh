#!/usr/bin/env bash
# Do datasheet estimates work better for big models? On a rented multi-GPU machine: probes of a
# ~310M model and of a 7B-shaped one (2 of its 32 layers timed and scaled), then short real DDP runs
# of the ~310M model (bench/backtest_probe.py --machine <out-dir> compares them).
#
# Usage:  bench/run_large_check.sh <lengths-dir> <out-dir> [number-of-gpus]
# QUICK=1 runs a few steps of everything, to test the script itself.
set -euo pipefail
DATA=${1:?usage: bench/run_large_check.sh <lengths-dir> <out-dir> [gpus]}
OUT=${2:?usage: bench/run_large_check.sh <lengths-dir> <out-dir> [gpus]}
GPUS=${3:-$(nvidia-smi -L | wc -l)}
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
T() { torchrun --standalone --nproc_per_node="$GPUS" "$@"; }
if [ "${QUICK:-0}" = "1" ]; then QD="--steps 8 --warmup 1"; QP="--rounds 1 --reps 1 --max-tokens 4096 --power-seconds 1"
else QD=""; QP=""; fi
ddp() { T bench/ddp_batches.py --out-dir "$OUT" --model large --repeats 1 "$@" $QD || echo "== FAILED: $*"; }
probe() { T -m evencut.train probe --max-len 2048 --max-tokens 16384 --rounds 2 "$@" $QP || echo "== FAILED: probe $*"; }
stamp() { echo "== $(date +%H:%M:%S) $*"; }
ROWS="noise floor;random | equal counts;random | Evencut, balanced padded size;whole optimizer step"

nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv
stamp probe 310M model
probe --layers 24 --d-model 1024 --vocab 8192 -o "$OUT/probe_large.json"
stamp probe 7B shape
probe --layers 32 --d-model 4096 --vocab 32000 --probe-layers 2 -o "$OUT/probe_7b.json"
stamp TinyStories 310M accumulation 4
ddp --lengths "$DATA/tinystories_tr.txt" --offset 20480 --global-batch $((16 * GPUS)) --accum 4 --steps 24 --warmup 4 \
    --extended --only "$ROWS" --name TinyStories-Turkish --out-name ddp_tinystories_large_accum4
stamp Wikipedia 310M
ddp --lengths "$DATA/wikipedia_tr_run.txt" --seq-len 2048 --global-batch $((6 * GPUS)) --steps 24 --warmup 3 \
    --only "$ROWS" --name Wikipedia-TR --out-name ddp_wikipedia_large
stamp ALLDONE
