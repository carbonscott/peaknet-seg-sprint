#!/bin/bash
# Wait for a training srun (PID on the bridge host) to exit, then run hf_eval_pipeline.sh in a GPU step.
# usage (from $PK/work/<agent>, detached): bash scripts/hf_chain_eval.sh <pid> <gpu> <agent> <model> <exp> [eval args]
P=$1; G=$2; AG=$3; M=$4; EXP=$5; shift 5
D=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
TAG=${EVAL_TAG:-test}
while kill -0 $P 2>/dev/null; do sleep 30; done
mkdir -p $D/evals/$AG/$EXP
echo "[chain] training PID $P exited at $(date); starting eval" > $D/evals/$AG/$EXP/eval_$TAG.log
srun --jobid=${JOBID:-47394816} --overlap --ntasks=1 --cpus-per-task=6 --mem=24G --gres=gpu:l40s:10 \
  bash -c "export CUDA_VISIBLE_DEVICES=$G EVAL_TAG=$TAG; bash scripts/hf_eval_pipeline.sh $AG $M $EXP $*" \
  >> $D/evals/$AG/$EXP/eval_$TAG.log 2>&1
