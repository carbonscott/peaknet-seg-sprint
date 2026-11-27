#!/bin/bash
# Whole-test eval of a VAL-picked (checkpoint, thr, tta) for hf_upernet as one detached srun step.
# usage (sdfiana025): bash scripts/hf_upernet_testonly.sh <jobid> <gpu> <exp> <best|last> <thr> <tta 0|1> <config>
# The pick comes from $D/evals/h4/<exp>/valgrid_test_{best,last}.json (scripts/val_grid.py on 400 val images).
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
D=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
J=$1; G=$2; EXP=$3; C=$4; T=$5; TT=$6; CFG=$7
R=$PK/runs/h4/$EXP; EV=$D/evals/h4/$EXP; CK=$D/checkpoints/h4/$EXP
mkdir -p $EV $CK $D/figures/h4/$EXP/test $D/logs/h4/$EXP
cp -f $R/$C.pt $CK/best.pt; cp -f $R/train.log $R/metrics.jsonl $D/logs/h4/$EXP/
[ "$TT" = 1 ] && TTA=--tta || TTA=
CMD="python -m seg.evaluate --model hf_upernet --config $CFG --ckpt $CK/best.pt --split test --thr $T $TTA --out $EV/metrics_test.json --save-overlays $D/figures/h4/$EXP/test --n-overlays 6 --workers 5"
echo "PK_MASK_SATURATED=0 $CMD" > $EV/eval_cmd_test.txt
cd $PK/work/h4
setsid nohup srun --jobid=$J --overlap --ntasks=1 --cpus-per-task=6 --mem=20G --gres=gpu:l40s:10 \
  bash -c "source $PK/env/venv/bin/activate; export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$G PK_MASK_SATURATED=0; \
  hostname; date; echo VAL_PICK ckpt=$C.pt thr=$T tta=$TT; echo $CMD; $CMD; date" > $EV/eval_test.log 2>&1 < /dev/null &
echo started test-only eval $EXP
