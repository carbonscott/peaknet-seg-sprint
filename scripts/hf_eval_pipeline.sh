#!/bin/bash
# B-hf crew: copy a finished seg.train run to durable storage, pick checkpoint AND threshold on
# the WHOLE val split, then evaluate the whole test split once at that (ckpt, thr), with overlays.
#   candidates: best.pt (seg.train's pick, val sweep may be capped at 0.9) and last.pt
#   VAL_THRS: thresholds swept on val
# Run inside an srun GPU step from $PK/work/<agent>:
#   bash scripts/hf_eval_pipeline.sh <agent> <model> <exp> [extra seg.evaluate args, e.g. --tta]
set -uo pipefail
AG=$1; M=$2; EXP=$3; shift 3
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
D=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
RUN=$PK/runs/$AG/$EXP
TAG=${EVAL_TAG:-test}
CFG=${EVAL_CFG:-configs/$M.yaml}
VAL_THRS=${VAL_THRS:-0.5,0.8,0.9,0.95,0.97,0.98,0.99,0.995,0.998,0.999}
CK=$D/checkpoints/$AG/$EXP; EV=$D/evals/$AG/$EXP
mkdir -p $CK $D/logs/$AG/$EXP $EV $D/figures/$AG/$EXP/$TAG
cp $RUN/train.log $RUN/metrics.jsonl $D/logs/$AG/$EXP/
source $PK/env/venv/bin/activate
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1
SPLIT=results/split/result.json
[ -f $SPLIT ] || SPLIT=$D/logs/h1/split/split.json
echo "[pipeline] $(date) host $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES step ${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-}"
BEST_F=-1; BEST_C=; BEST_T=
for c in best last; do
  [ -f $RUN/$c.pt ] || continue
  python -u -m seg.evaluate --model $M --config $CFG --ckpt $RUN/$c.pt --split val --split-json $SPLIT \
    --workers 6 --thr $VAL_THRS --out $EV/metrics_val_${TAG}_$c.json "$@"
  read F T < <(python -c "import json;d=json.load(open('$EV/metrics_val_${TAG}_$c.json'));print(d['peak']['f1'],d['threshold'])")
  echo "[pipeline] val $c.pt: peak F1 $F at thr $T"
  if python -c "import sys;sys.exit(0 if $F > $BEST_F else 1)"; then BEST_F=$F; BEST_C=$c; BEST_T=$T; fi
done
echo "[pipeline] selected $BEST_C.pt thr $BEST_T (val peak F1 $BEST_F) -> $CK/best.pt"
cp $RUN/$BEST_C.pt $CK/best.pt
cp $EV/metrics_val_${TAG}_$BEST_C.json $EV/metrics_val_$TAG.json
set -x
python -u -m seg.evaluate --model $M --config $CFG --ckpt $CK/best.pt \
  --split test --split-json $SPLIT --workers 6 --out $EV/metrics_$TAG.json \
  --save-overlays $D/figures/$AG/$EXP/$TAG --n-overlays 6 --thr $BEST_T "$@"
echo "[pipeline] done $(date) rc=$?"
