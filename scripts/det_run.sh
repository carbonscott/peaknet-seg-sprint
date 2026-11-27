#!/bin/bash
# One Slurm step: train the RT-DETRv2 peak detector, evaluate the whole test split, copy durable outputs.
# usage (inside srun, from work/<you>): bash scripts/det_run.sh <exp> [config]
set -uo pipefail
EXP=$1; CFG=${2:-configs/detection.yaml}
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
DUR=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
RUN=$PK/runs/h8/$EXP
source $PK/env/venv/bin/activate
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=4
mkdir -p $RUN $DUR/checkpoints/h8/$EXP $DUR/logs/h8/$EXP $DUR/evals/h8/$EXP $DUR/figures/h8/$EXP
echo "step SLURM_JOB_ID=$SLURM_JOB_ID SLURM_STEP_ID=$SLURM_STEP_ID host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES start $(date)"
python -m seg.detection train --config $CFG --out $RUN
cp $RUN/best.pt $DUR/checkpoints/h8/$EXP/best.pt; cp $RUN/train.log $DUR/logs/h8/$EXP/train.log
python -m seg.detection eval --config $CFG --ckpt $DUR/checkpoints/h8/$EXP/best.pt --split test \
    --out $DUR/evals/h8/$EXP/test_metrics.json --fig-dir $DUR/figures/h8/$EXP --fig-every 150 2>&1 | grep -v -e "^- model" -e Warning > $DUR/evals/h8/$EXP/eval.log
echo "end $(date)"
