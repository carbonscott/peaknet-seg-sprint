#!/bin/bash
# usage (on sdfiana025 via bridge): bash scripts/hf_upernet_eval.sh <jobid> <gpu> <exp> [extra seg.evaluate args, e.g. --tta]
# 1) copies best.pt, train.log, metrics.jsonl of $PK/runs/h4/<exp> to the durable sprint dirs
# 2) runs seg.evaluate on the WHOLE test split from the durable checkpoint (threshold = val-picked thr in best.pt)
#    as a detached srun step; metrics JSON, eval log and overlay PNGs go to the durable dirs.
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
D=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
J=$1; G=$2; EXP=$3; shift 3
TAG=${EVAL_TAG:-test}
R=$PK/runs/h4/$EXP
mkdir -p $D/checkpoints/h4/$EXP $D/evals/h4/$EXP $D/figures/h4/$EXP/$TAG $D/logs/h4/$EXP
cp -f $R/best.pt $D/checkpoints/h4/$EXP/best.pt
cp -f $R/train.log $R/metrics.jsonl $D/logs/h4/$EXP/ 2>/dev/null
cd $PK/work/h4
# optional: VAL_SWEEP=1 -> pick threshold on the first ${VAL_N:-500} VAL images over a wide sweep, then test at that thr
VCMD="python -m seg.evaluate --model hf_upernet --config configs/hf_upernet.yaml --ckpt $D/checkpoints/h4/$EXP/best.pt --split val --limit ${VAL_N:-500} --thr 0.5,0.7,0.8,0.9,0.93,0.95,0.97,0.98,0.99,0.995,0.998,0.999 --out $D/evals/h4/$EXP/metrics_valsweep_$TAG.json --workers 6"
CMD="python -m seg.evaluate --model hf_upernet --config configs/hf_upernet.yaml --ckpt $D/checkpoints/h4/$EXP/best.pt --split test --out $D/evals/h4/$EXP/metrics_$TAG.json --save-overlays $D/figures/h4/$EXP/$TAG --n-overlays 6 --workers 6 $*"
echo "$CMD" > $D/evals/h4/$EXP/eval_cmd_$TAG.txt
setsid nohup srun --jobid=$J --overlap --ntasks=1 --cpus-per-task=7 --mem=40G --gres=gpu:l40s:10 \
  bash -c "source $PK/env/venv/bin/activate; export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$G; \
  hostname; date; if [ -n \"$VAL_SWEEP\" ]; then echo $VCMD; $VCMD; T=\$(python -c \"import json;print(json.load(open('$D/evals/h4/$EXP/metrics_valsweep_$TAG.json'))['threshold'])\"); echo VAL_PICKED_THR=\$T; else T=; fi; \
  echo $CMD \${T:+--thr \$T}; $CMD \${T:+--thr \$T}; date" > $D/evals/h4/$EXP/eval_$TAG.log 2>&1 < /dev/null &
echo started eval $EXP $TAG
