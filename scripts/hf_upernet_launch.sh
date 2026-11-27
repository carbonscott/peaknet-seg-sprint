#!/bin/bash
# usage (on sdfiana025 via bridge): bash scripts/hf_upernet_launch.sh <jobid> <gpus e.g. 0 or 0,1> <exp> [seg.train args...]
# Runs seg.train for hf_upernet as a detached srun step; log at $PK/runs/h4/<exp>/train.log
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
J=$1; G=$2; EXP=$3; shift 3
NG=$(echo $G | tr ',' '\n' | wc -l)
R=$PK/runs/h4/$EXP; mkdir -p $R
cd $PK/work/h4
if [ "$NG" -gt 1 ]; then RUN="torchrun --standalone --nproc_per_node=$NG -m seg.train"; else RUN="python -m seg.train"; fi
setsid nohup srun --jobid=$J --overlap --ntasks=1 --cpus-per-task=$((10*NG)) --mem=$((90*NG))G --gres=gpu:l40s:10 \
  bash -c "source $PK/env/venv/bin/activate; export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$G; \
  hostname; nvidia-smi -L; $RUN --model hf_upernet --out $R $*" > $R/train.log 2>&1 < /dev/null &
echo started $R
