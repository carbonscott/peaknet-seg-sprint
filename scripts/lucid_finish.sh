#!/bin/bash
# Copy a finished run to durable storage and evaluate its best.pt on the whole test split.
# usage (inside an srun step, from $PK/work/<you>): bash scripts/lucid_finish.sh <model> <exp> [extra seg.evaluate args]
# Durable layout (boss order): results/peaknet-seg-sprint/{checkpoints,logs,evals,figures}/<agent>/<exp>/
set -euo pipefail
MODEL=$1; EXP=$2; shift 2
AG=${AGENT:-h5}
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
DUR=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
RUN=$PK/runs/$AG/$EXP
# wide val threshold sweep (overrides any earlier --val-thr; argparse keeps the last one)
VAL_THR=${VAL_THR-0.7,0.9,0.95,0.98,0.99,0.995,0.998,0.999,0.9995,0.9998,0.9999}
mkdir -p $DUR/checkpoints/$AG/$EXP $DUR/logs/$AG/$EXP $DUR/evals/$AG/$EXP $DUR/figures/$AG/$EXP
cp $RUN/best.pt $DUR/checkpoints/$AG/$EXP/best.pt
cp $RUN/train.log $RUN/metrics.jsonl $DUR/logs/$AG/$EXP/
python -m seg.evaluate --model $MODEL --ckpt $DUR/checkpoints/$AG/$EXP/best.pt --split test \
    --out $DUR/evals/$AG/$EXP/metrics.json --save-overlays $DUR/figures/$AG/$EXP --n-overlays 6 "$@" ${VAL_THR:+--val-thr $VAL_THR} \
    2>&1 | tee $DUR/evals/$AG/$EXP/eval.log
echo FINISHED
