#!/bin/bash
# best_model final pipeline (h7). Run on the bridge host, detached:
#   setsid nohup bash scripts/best_final.sh <tag> > <log> 2>&1 < /dev/null &
# 1) waits until smp_r34_a and smp_upp34_a training logs show the "end" event
# 2) copies best.pt / last.pt / train.log / metrics.jsonl to the durable sprint dirs
# 3) ONE GPU step: seg.evaluate_ensemble picks member subset + threshold on VAL (flip TTA), then whole TEST
set -u
TAG=${1:-final1}
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
S=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
RUNS="smp_r34_a smp_upp34_a"
for r in $RUNS; do
  until grep -q '"event": "end"' $PK/runs/h7/$r/train.log 2>/dev/null; do sleep 30; done
  echo "$(date) $r finished"
  mkdir -p $S/checkpoints/h7/$r $S/logs/h7/$r
  cp $PK/runs/h7/$r/best.pt $S/checkpoints/h7/$r/best.pt
  cp $PK/runs/h7/$r/last.pt $S/checkpoints/h7/$r/last.pt
  cp $PK/runs/h7/$r/train.log $PK/runs/h7/$r/metrics.jsonl $S/logs/h7/$r/
done
E=$S/evals/h7/best_model/$TAG; F=$S/figures/h7/best_model/$TAG; mkdir -p $E $F
C=$S/checkpoints/h7
cd $PK/work/h7
echo "$(date) launching eval step"
srun --jobid=47394816 --overlap --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:l40s:10 bash -c "
source $PK/env/venv/bin/activate; export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
python -m seg.evaluate_ensemble \
  --member smp_unet:$C/smp_r34_a/best.pt --member smp_unet:$C/smp_r34_a/last.pt \
  --member smp_unet:$C/smp_upp34_a/best.pt --member smp_unet:$C/smp_upp34_a/last.pt \
  --subsets '0;1;2;3;0,2;1,3;0,3;1,2;0,1,2,3' --tta --val-images 300 --procs 3 --workers 3 \
  --save-overlays $F --n-overlays 4 --out $E/test.json" > $E/eval.log 2>&1
echo "$(date) eval done rc=$?"
