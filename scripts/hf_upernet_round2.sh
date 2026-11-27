#!/bin/bash
# Round-2 chain (run detached on sdfiana025): wait for upn_a test metrics, then fine-tune upn_ft from the
# val-picked upn_a checkpoint with configs/hf_upernet_ft.yaml, then VAL-pick + whole-test eval of upn_ft.
# usage: bash scripts/hf_upernet_round2.sh <jobid> <gpu> <time_min>
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
D=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
J=$1; G=$2; TM=$3
cd $PK/work/h4
until [ -f $D/evals/h4/upn_a/metrics_test.json ]; do sleep 30; done
sleep 20; date; echo "upn_a eval done; launching upn_ft"
bash scripts/hf_upernet_launch.sh $J $G upn_ft --config configs/hf_upernet_ft.yaml --init-ckpt $D/checkpoints/h4/upn_a/best.pt --time-min $TM
sleep 10
P=$(ps -u $USER -o pid,args | grep "[s]run --jobid=$J" | grep "runs/h4/upn_ft" | head -1 | awk '{print $1}')
echo "upn_ft srun pid $P"; while kill -0 $P 2>/dev/null; do sleep 30; done
date; echo "upn_ft training done; launching eval"
CONFIG=configs/hf_upernet_ft.yaml bash scripts/hf_upernet_eval.sh $J $G upn_ft
