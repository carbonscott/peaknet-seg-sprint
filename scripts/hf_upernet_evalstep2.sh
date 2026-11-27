#!/bin/bash
# Runs INSIDE the srun step started by scripts/hf_upernet_eval.sh (cwd = $PK/work/h4, venv active).
# usage: hf_upernet_evalstep.sh <run dir> <durable root D> <exp> <tag>   (env: CONFIG, VAL_N)
# 1) For each candidate checkpoint (best.pt = in-run val pick, last.pt = end of schedule) run scripts/val_grid.py on
#    $VAL_N evenly spaced VAL images: flip-TTA x saturated-pixel mask (0/1/2 px) x threshold 0.5..0.999.
#    (The in-run pick only sweeps up to train.thresholds, so a later checkpoint can win at a higher threshold.)
# 2) Pick (checkpoint, thr) [tta/mask only if ALLOW_TTA=1; default = best no-TTA, no-mask row] with the best VAL peak F1; copy that checkpoint to $D/checkpoints/h4/<exp>/best.pt.
# 3) Evaluate the WHOLE test split once with that pick (+ overlays). Nothing on test is used for selection.
set -u
R=$1; D=$2; EXP=$3; TAG=$4; shift 4
EV=$D/evals/h4/$EXP; CK=$D/checkpoints/h4/$EXP
mkdir -p $EV $CK $D/figures/h4/$EXP/$TAG
CFG=${CONFIG:-configs/hf_upernet.yaml}
for c in best last; do
  [ -f $R/$c.pt ] || continue
  echo "== VAL grid $c.pt (scripts/val_grid.py: flip-TTA x saturated mask 0/1/2 x thr .5-.999)"; date
  python scripts/val_grid.py --member hf_upernet:$R/$c.pt:$CFG --n ${VAL_N:-400} --masks 0 --min-sizes 0 --thrs 0.5,0.7,0.8,0.9,0.95,0.97,0.98,0.99,0.995,0.998,0.999,0.9995,0.9998,0.9999 \
    --out $EV/valgrid_${TAG}_$c.json --workers 3 --procs 4 2>&1 | grep -v "^\[val_grid\]" | tail -3
done
PICK=$(python - "$EV" "$TAG" <<'EOF2'
import json, os, sys
ev, tag = sys.argv[1:3]
best = None
for c in ("best", "last"):
    p = "%s/valgrid_%s_%s.json" % (ev, tag, c)
    if os.path.isfile(p):
        g = json.load(open(p)); r = g["best_evaluate_reachable"]; b0 = g["baseline_no_tta_no_mask"]
        print("VAL %s: best %s | baseline(no tta, no mask) thr %s F1 %.4f" % (c, r, b0["thr"], b0["peak_f1"]), file=sys.stderr)
        if not os.environ.get("ALLOW_TTA"):
            r = b0  # no flip-TTA on test (4x inference cost); mask_saturated had no effect on val for this model
        if best is None or r["peak_f1"] > best["peak_f1"]:
            best = dict(r, ckpt=c, n_val_images=g["n_images"])
json.dump(best, open("%s/val_pick_%s.json" % (ev, tag), "w"))
print(best["ckpt"], best["thr"], best["mask_saturated"], "--tta" if best["tta"] else "")
EOF2
)
set -- $PICK; C=$1; T=$2; MS=$3; TTA=${4:-}
echo "VAL_PICK ckpt=$C.pt thr=$T PK_MASK_SATURATED=$MS tta=$TTA"
export PK_MASK_SATURATED=$MS
cp -f $R/$C.pt $CK/best.pt
CMD="python -m seg.evaluate --model hf_upernet --config $CFG --ckpt $CK/best.pt --split test --thr $T $TTA --out $EV/metrics_$TAG.json --save-overlays $D/figures/h4/$EXP/$TAG --n-overlays 6 --workers 6 ${TEST_EXTRA:-}"
echo "PK_MASK_SATURATED=$MS $CMD" > $EV/eval_cmd_$TAG.txt
echo "== TEST: $CMD"; date
$CMD
date
