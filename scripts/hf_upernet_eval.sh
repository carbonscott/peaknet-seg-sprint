#!/bin/bash
# usage (on sdfiana025 via bridge): bash scripts/hf_upernet_eval.sh <jobid> <gpu> <exp> [extra seg.evaluate args, e.g. --tta]
# Copies train.log + metrics.jsonl of $PK/runs/h4/<exp> to the durable sprint dirs, then runs
# scripts/hf_upernet_evalstep.sh (VAL pick of checkpoint+threshold, then WHOLE test eval) as one detached srun step.
# Log: $D/evals/h4/<exp>/eval_<tag>.log ; metrics: $D/evals/h4/<exp>/metrics_<tag>.json
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
D=${DURABLE:-/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint}
J=$1; G=$2; EXP=$3; shift 3
TAG=${EVAL_TAG:-test}
R=$PK/runs/h4/$EXP
mkdir -p $D/checkpoints/h4/$EXP $D/evals/h4/$EXP $D/figures/h4/$EXP/$TAG $D/logs/h4/$EXP
cp -f $R/train.log $R/metrics.jsonl $D/logs/h4/$EXP/ 2>/dev/null
cd $PK/work/h4
setsid nohup srun --jobid=$J --overlap --ntasks=1 --cpus-per-task=7 --mem=40G --gres=gpu:l40s:10 \
  bash -c "source $PK/env/venv/bin/activate; export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$G; \
  hostname; bash scripts/hf_upernet_evalstep.sh $R $D $EXP $TAG $*" > $D/evals/h4/$EXP/eval_$TAG.log 2>&1 < /dev/null &
echo started eval $EXP $TAG
