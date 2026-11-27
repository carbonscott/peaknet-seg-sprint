#!/bin/bash
# best_model final2 (h7): adds h1's smp Unet efficientnet-b4 (run smp_effb4) to the subset search.
# Waits for the effb4 training "end" event, copies its best/last ckpts (read-only use) to h7's durable dir,
# then ONE GPU step: subset + thr on VAL (flip TTA, 200 imgs), whole TEST with the winner, test probs saved.
set -u
TAG=${1:-final2}
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
S=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
H1=$PK/runs/h1/smp_effb4
until grep -q '"event": "end"' $H1/train.log 2>/dev/null; do sleep 20; done
echo "$(date) effb4 finished"
M=$S/checkpoints/h7/ens_members; mkdir -p $M
cp $H1/best.pt $M/effb4_best.pt; cp $H1/last.pt $M/effb4_last.pt
C=$S/checkpoints/h7
E=$S/evals/h7/best_model/$TAG; F=$S/figures/h7/best_model/$TAG; mkdir -p $E $F
cd $PK/work/h7
echo "$(date) launching eval step"
srun --jobid=47394816 --overlap --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:l40s:10 bash -c "
source $PK/env/venv/bin/activate; export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
python -m seg.evaluate_ensemble \
  --member smp_unet:$C/smp_r34_a/last.pt --member smp_unet:$C/smp_upp34_a/best.pt \
  --member smp_unet:$M/effb4_best.pt --member smp_unet:$M/effb4_last.pt \
  --subsets '1;2;3;0,1;1,2;1,3;0,1,2;0,1,3;1,2,3;0,1,2,3' --tta --val-images 200 --procs 4 --workers 3 \
  --save-overlays $F --n-overlays 4 --save-probs $PK/runs/h7/final/test_probs_$TAG --out $E/test.json" > $E/eval.log 2>&1
echo "$(date) eval done rc=$?"
