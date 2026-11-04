#!/usr/bin/env bash
# Phase-3 check on a rented multi-GPU machine: the probe first, then the real DDP runs whose step
# times it should predict (bench/backtest_probe.py --machine <out-dir> compares them).
#
# Usage:  bench/run_probe_check.sh <lengths-dir> <out-dir> [number-of-gpus]
#
# <lengths-dir> holds tinystories_tr.txt, dolly15k_gpt2.txt and wikipedia_tr_run.txt (one length per
# line, already shuffled). Per-GPU work matches the earlier 8x RTX 4090 / 4x A100 runs. The most
# useful parts run first, so a run cut short by a time limit still leaves the probes behind.
set -euo pipefail
DATA=${1:?usage: bench/run_probe_check.sh <lengths-dir> <out-dir> [gpus]}
OUT=${2:?usage: bench/run_probe_check.sh <lengths-dir> <out-dir> [gpus]}
GPUS=${3:-$(nvidia-smi -L | wc -l)}
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
T() { torchrun --standalone --nproc_per_node="$GPUS" "$@"; }
# QUICK=1: a few steps of everything, to test the script itself
if [ "${QUICK:-0}" = "1" ]; then QD="--steps 8 --warmup 1 --repeats 1"; QP="--rounds 1 --reps 1 --max-tokens 4096 --power-seconds 1"; else QD=""; QP=""; fi
# a failed run (say, out of memory) is reported and the next one still runs
ddp() { T bench/ddp_batches.py --out-dir "$OUT" "$@" $QD || echo "== FAILED: $*"; }
stamp() { echo "== $(date +%H:%M:%S) $*"; }

nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv
stamp dry run
ddp --lengths "$DATA/tinystories_tr.txt" --offset 20480 --global-batch $((32 * GPUS)) --steps 3 --warmup 1 \
    --repeats 1 --name dry-run --out-name dry_run

stamp probe small model
T -m evencut.train probe --layers 8 --d-model 512 --vocab 8192 --max-len 2048 --max-tokens 16384 -o "$OUT/probe_small.json" $QP || echo "== FAILED: probe small"
stamp probe base model
T -m evencut.train probe --layers 12 --d-model 768 --vocab 8192 --max-len 2048 --max-tokens 16384 -o "$OUT/probe_base.json" $QP || echo "== FAILED: probe base"

stamp TinyStories small
ddp --lengths "$DATA/tinystories_tr.txt" --offset 20480 --global-batch $((32 * GPUS)) --name TinyStories-Turkish \
    --out-name ddp_tinystories
stamp Dolly small accumulation 8
ddp --lengths "$DATA/dolly15k_gpt2.txt" --seq-len 1024 --global-batch $((16 * GPUS)) --accum 8 --steps 80 --extended \
    --name Dolly-15k --out-name ddp_dolly_accum8
stamp Dolly base accumulation 4
ddp --lengths "$DATA/dolly15k_gpt2.txt" --seq-len 1024 --global-batch $((24 * GPUS)) --accum 4 --model base \
    --name Dolly-15k --out-name ddp_dolly_base_accum4
stamp Wikipedia small
ddp --lengths "$DATA/wikipedia_tr_run.txt" --seq-len 2048 --global-batch $((6 * GPUS)) --name Wikipedia-TR \
    --out-name ddp_wikipedia
stamp ALLDONE
