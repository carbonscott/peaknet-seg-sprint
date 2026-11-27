#!/bin/bash
# B-hf crew: copy a finished seg.train run to durable storage, evaluate best.pt on the WHOLE
# test split at a threshold picked by a sweep on the whole val split (VAL_THRS=none: use the thr in best.pt).
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
THR_ARGS=()
VAL_THRS=${VAL_THRS:-0.5,0.7,0.8,0.85,0.9,0.92,0.94,0.95,0.96,0.97,0.98,0.985,0.99,0.995}
if [ "$VAL_THRS" != "none" ]; then
  # pick the threshold on the WHOLE val split (never on test)
  python -u -m seg.evaluate --model $M --config configs/$M.yaml --ckpt $D/checkpoints/$AG/$EXP/best.pt \
    --split val --split-json $SPLIT --workers 6 --thr $VAL_THRS --out $D/evals/$AG/$EXP/metrics_val_$TAG.json "$@"
  T=$(python -c "import json;print(json.load(open('$D/evals/$AG/$EXP/metrics_val_$TAG.json'))['threshold'])")
  echo "[pipeline] val-picked threshold $T"
  THR_ARGS=(--thr $T)
fi
set -x
python -u -m seg.evaluate --model $M --config configs/$M.yaml --ckpt $D/checkpoints/$AG/$EXP/best.pt \
  --split test --split-json $SPLIT --workers 6 --out $D/evals/$AG/$EXP/metrics_$TAG.json \
  --save-overlays $D/figures/$AG/$EXP/$TAG --n-overlays 6 "${THR_ARGS[@]}" "$@"
echo "[pipeline] done $(date) rc=$?"
