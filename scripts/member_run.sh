#!/bin/bash
# One job step: train -> val_grid (TTA x saturated mask x thr) on best.pt and last.pt -> whole-test seg.evaluate
# of the better one at its val-picked settings -> durable copies.
# usage (from $PK/work/<you>, inside srun with CUDA_VISIBLE_DEVICES set):
#   bash scripts/member_run.sh <model> <config> <exp> <time_min> [val_n=300]
set -uo pipefail
MODEL=$1; CFG=$2; EXP=$3; TMIN=$4; VN=${5:-300}
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
DUR=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
ME=$(basename "$PWD")
R=$PK/runs/$ME/$EXP
mkdir -p "$R" "$DUR/checkpoints/$ME/$EXP" "$DUR/logs/$ME/$EXP" "$DUR/evals/$ME/$EXP" "$DUR/figures/$ME/$EXP"
source $PK/env/venv/bin/activate
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1
echo "[member] $(date) host $(hostname) cvd=$CUDA_VISIBLE_DEVICES step=${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-} model=$MODEL cfg=$CFG"
python -u -m seg.train --model "$MODEL" --config "$CFG" --out "$R" --time-min "$TMIN"
echo "[member] train exit $? at $(date)"
[ -f "$R/best.pt" ] || { echo "[member] no best.pt: abort"; exit 1; }
cp "$R/metrics.jsonl" "$DUR/logs/$ME/$EXP/metrics.jsonl"
for c in best last; do
  [ -f "$R/$c.pt" ] || continue
  python -u scripts/val_grid.py --member "$MODEL:$R/$c.pt:$CFG" --n "$VN" --workers 3 --procs 6 \
    --out "$R/grid_$c.json" > "$R/grid_$c.log" 2>&1
  echo "[member] val_grid $c exit $? at $(date)"
done
PICK=$(python - "$R" <<'PY'
import json, os, sys
R = sys.argv[1]
best = None
for c in ("best", "last"):
    p = os.path.join(R, "grid_%s.json" % c)
    if os.path.isfile(p):
        g = json.load(open(p))["best_evaluate_reachable"]
        if best is None or g["peak_f1"] > best[1]["peak_f1"]:
            best = (c, g)
c, g = best
print(c, g["thr"], int(g["mask_saturated"]), int(bool(g["tta"])), round(g["peak_f1"], 4))
PY
)
read C THR MS TTA VF1 <<< "$PICK"
echo "[member] pick ckpt=$C thr=$THR mask_saturated=$MS tta=$TTA val_f1=$VF1"
cp "$R/$C.pt" "$DUR/checkpoints/$ME/$EXP/best.pt"
TT=""; [ "$TTA" = "1" ] && TT="--tta"
PK_MASK_SATURATED=$MS python -u -m seg.evaluate --model "$MODEL" --config "$CFG" --ckpt "$DUR/checkpoints/$ME/$EXP/best.pt" \
  --split test --split-json results/split/result.json --peak-label 1 --workers 8 --thr "$THR" $TT \
  --out "$R/test_metrics.json" --save-overlays "$R/overlays" --n-overlays 4 > "$R/eval.log" 2>&1
echo "[member] test eval exit $? at $(date)"
tail -n 25 "$R/eval.log"
cp "$R"/grid_*.json "$R"/grid_*.log "$R/test_metrics.json" "$R/eval.log" "$DUR/evals/$ME/$EXP/"
cp -r "$R/overlays/." "$DUR/figures/$ME/$EXP/" 2>/dev/null
cp "$R/train.log" "$DUR/logs/$ME/$EXP/train.log"
echo "[member] done $(date)"
