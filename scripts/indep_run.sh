#!/bin/bash
# One Slurm step (h8): independent inference of best_model final1 members + independent scoring.
set -uo pipefail
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
DUR=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
CK=$DUR/checkpoints/h7
O=$DUR/evals/h8/indep_infer_final1; P=$PK/runs/h8/indep_probs_final1
source $PK/env/venv/bin/activate
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=4
mkdir -p $O $P $DUR/figures/h8/reeval
echo "step SLURM_JOB_ID=$SLURM_JOB_ID SLURM_STEP_ID=$SLURM_STEP_ID host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES start $(date)"
python scripts/independent_infer.py --member smp_unet:$CK/smp_r34_a/last.pt --member smp_unet:$CK/smp_upp34_a/best.pt --tta --out-probs $P 2>&1 | grep -v Warn
python scripts/independent_reeval.py --probs $P --thr 0.7 --split test --procs 4 --out $O/reeval.json \
  --overlays mfxl1025422_r0309_peaknet.0002.zarr__17 mfxl1027522_r0028_peaknet.0007.zarr__18 --fig-dir $DUR/figures/h8/reeval \
  --title "independent inference + scoring of best_model final1 (h8)"
echo "end $(date)"
