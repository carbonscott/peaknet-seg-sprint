#!/bin/bash
# Round 2 started early (03:20 PST) because the upn_a whole-test eval (4-flip val grid) runs long:
# fine-tune upn_ft from upn_a last.pt (end of its cosine schedule) on the same GPU, small step (6 CPU / 20G),
# then VAL-pick + whole-test eval of upn_ft with the same footprint.
# usage: bash scripts/hf_upernet_round2_now.sh <jobid> <gpu> <time_min>
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
J=$1; G=$2; TM=$3
cd $PK/work/h4
export STEP_CPUS=6 STEP_MEM=20G
bash scripts/hf_upernet_launch.sh $J $G upn_ft --config configs/hf_upernet_ft.yaml --init-ckpt $PK/runs/h4/upn_a/last.pt --time-min $TM --set train.num_workers=4
sleep 15
P=$(ps -u $USER -o pid,args | grep "[s]run --jobid=$J" | grep "runs/h4/upn_ft" | head -1 | awk '{print $1}')
echo "upn_ft srun pid $P"; while kill -0 $P 2>/dev/null; do sleep 30; done
date; echo "upn_ft training done; launching eval"
CONFIG=configs/hf_upernet_ft.yaml VAL_N=${VAL_N:-300} bash scripts/hf_upernet_eval.sh $J $G upn_ft
