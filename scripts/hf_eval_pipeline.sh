#!/bin/bash
# B-hf crew: copy a finished seg.train run to durable storage; pick checkpoint (best.pt vs last.pt),
# threshold, flip-TTA and saturated-pixel mask on VAL with scripts/val_grid.py (h1); then evaluate
# the WHOLE test split once with that pick (seg.evaluate, overlays).
# Run inside an srun GPU step from $PK/work/<agent>:
#   bash scripts/hf_eval_pipeline.sh <agent> <model> <exp>
# env: EVAL_TAG (default test), EVAL_CFG (default configs/<model>.yaml), VAL_N (default 400), GRID_TILE
set -uo pipefail
AG=$1; M=$2; EXP=$3; shift 3
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
D=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
RUN=$PK/runs/$AG/$EXP
TAG=${EVAL_TAG:-test}
CFG=${EVAL_CFG:-configs/$M.yaml}
VAL_N=${VAL_N:-400}
CK=$D/checkpoints/$AG/$EXP; EV=$D/evals/$AG/$EXP
mkdir -p $CK $D/logs/$AG/$EXP $EV $D/figures/$AG/$EXP/$TAG
cp $RUN/train.log $RUN/metrics.jsonl $D/logs/$AG/$EXP/
source $PK/env/venv/bin/activate
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1
SPLIT=results/split/result.json
[ -f $SPLIT ] || SPLIT=$D/logs/h1/split/split.json
echo "[pipeline] $(date) host $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES step ${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-}"
BEST_F=-1; BEST_C=
for c in best last; do
  [ -f $RUN/$c.pt ] || continue
  python -u scripts/val_grid.py --member $M:$RUN/$c.pt:$CFG --n $VAL_N --masks 0,1,2 --min-sizes 0 \
    ${GRID_TILE:+--tile $GRID_TILE} --out $EV/val_grid_${TAG}_$c.json > /dev/null
  F=$(python -c "import json;print(json.load(open('$EV/val_grid_${TAG}_$c.json'))['best_evaluate_reachable']['peak_f1'])")
  echo "[pipeline] val grid $c.pt: best reachable peak F1 $F"
  if python -c "import sys;sys.exit(0 if $F > $BEST_F else 1)"; then BEST_F=$F; BEST_C=$c; fi
done
cp $EV/val_grid_${TAG}_$BEST_C.json $EV/val_grid_$TAG.json
read T TTA MS < <(python -c "import json;b=json.load(open('$EV/val_grid_$TAG.json'))['best_evaluate_reachable'];print(b['thr'],int(b['tta']),b['mask_saturated'])")
echo "[pipeline] selected $BEST_C.pt thr $T tta $TTA mask_saturated $MS (val peak F1 $BEST_F) -> $CK/best.pt"
cp $RUN/$BEST_C.pt $CK/best.pt
TTA_ARG=; [ "$TTA" = 1 ] && TTA_ARG=--tta
set -x
PK_MASK_SATURATED=$MS python -u -m seg.evaluate --model $M --config $CFG --ckpt $CK/best.pt \
  --split test --split-json $SPLIT --workers 6 --out $EV/metrics_$TAG.json \
  --save-overlays $D/figures/$AG/$EXP/$TAG --n-overlays 6 --thr $T $TTA_ARG "$@"
echo "[pipeline] done $(date) rc=$?"
