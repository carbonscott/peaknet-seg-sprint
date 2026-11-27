#!/bin/bash
# B-hf crew: copy a finished seg.train run to durable storage, evaluate best.pt on the WHOLE
# test split (threshold = the val-picked thr stored in best.pt), save overlays.
# Run inside an srun GPU step from $PK/work/<agent>:
#   bash scripts/hf_eval_pipeline.sh <agent> <model> <exp> [extra seg.evaluate args, e.g. --tta]
set -uo pipefail
AG=$1; M=$2; EXP=$3; shift 3
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
D=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
RUN=$PK/runs/$AG/$EXP
TAG=${EVAL_TAG:-test}
mkdir -p $D/checkpoints/$AG/$EXP $D/logs/$AG/$EXP $D/evals/$AG/$EXP $D/figures/$AG/$EXP/$TAG
cp -u $RUN/best.pt $D/checkpoints/$AG/$EXP/best.pt
cp $RUN/train.log $RUN/metrics.jsonl $D/logs/$AG/$EXP/
source $PK/env/venv/bin/activate
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1
SPLIT=results/split/result.json
[ -f $SPLIT ] || SPLIT=$D/logs/h1/split/split.json
echo "[pipeline] $(date) host $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES step ${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-}"
set -x
python -u -m seg.evaluate --model $M --config configs/$M.yaml --ckpt $D/checkpoints/$AG/$EXP/best.pt \
  --split test --split-json $SPLIT --workers 6 --out $D/evals/$AG/$EXP/metrics_$TAG.json \
  --save-overlays $D/figures/$AG/$EXP/$TAG --n-overlays 6 "$@"
echo "[pipeline] done $(date) rc=$?"
