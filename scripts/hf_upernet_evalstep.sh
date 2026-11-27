#!/bin/bash
# Runs INSIDE the srun step started by scripts/hf_upernet_eval.sh (cwd = $PK/work/h4, venv active).
# usage: hf_upernet_evalstep.sh <run dir> <durable root D> <exp> <tag> [extra seg.evaluate args for the test pass]
# 1) For each candidate checkpoint in the run (best.pt = in-run val pick, last.pt = end of schedule), sweep the
#    threshold on the first $VAL_N VAL images (0.5..0.999). The in-run pick only sweeps up to train.thresholds,
#    so a later checkpoint can be better at a higher threshold.
# 2) Pick (checkpoint, threshold) with the best VAL peak F1; copy that checkpoint to $D/checkpoints/h4/<exp>/best.pt.
# 3) Evaluate the WHOLE test split once at that threshold (+ overlays). Nothing on test is used for selection.
set -u
R=$1; D=$2; EXP=$3; TAG=$4; shift 4
EV=$D/evals/h4/$EXP; CK=$D/checkpoints/h4/$EXP
mkdir -p $EV $CK $D/figures/h4/$EXP/$TAG
THRS=${THRS:-0.5,0.7,0.8,0.9,0.93,0.95,0.97,0.98,0.99,0.995,0.998,0.999}
for c in best last; do
  [ -f $R/$c.pt ] || continue
  echo "== VAL sweep $c.pt"; date
  python -m seg.evaluate --model hf_upernet --config configs/hf_upernet.yaml --ckpt $R/$c.pt --split val \
    --limit ${VAL_N:-500} --thr $THRS --out $EV/valsweep_${TAG}_$c.json --workers 6 2>&1 | grep -v "^\[evaluate\] *[0-9]*/" | tail -5
done
PICK=$(python - "$EV" "$TAG" <<'EOF'
import json, os, sys
ev, tag = sys.argv[1:3]
best = None
for c in ("best", "last"):
    p = "%s/valsweep_%s_%s.json" % (ev, tag, c)
    if os.path.isfile(p):
        m = json.load(open(p))
        print("VAL %s: thr %s peak F1 %.4f P %.4f R %.4f n %d" % (c, m["threshold"], m["peak"]["f1"], m["peak"]["precision"],
              m["peak"]["recall"], m["n_images"]), file=sys.stderr)
        if best is None or m["peak"]["f1"] > best[2]:
            best = (c, m["threshold"], m["peak"]["f1"])
json.dump({"ckpt": best[0], "thr": best[1], "val_peak_f1": best[2]}, open("%s/val_pick_%s.json" % (ev, tag), "w"))
print(best[0], best[1])
EOF
)
set -- $PICK "$@"; C=$1; T=$2; shift 2
echo "VAL_PICK ckpt=$C.pt thr=$T"
cp -f $R/$C.pt $CK/best.pt
CMD="python -m seg.evaluate --model hf_upernet --config configs/hf_upernet.yaml --ckpt $CK/best.pt --split test --thr $T --out $EV/metrics_$TAG.json --save-overlays $D/figures/h4/$EXP/$TAG --n-overlays 6 --workers 6 $*"
echo "$CMD" > $EV/eval_cmd_$TAG.txt
echo "== TEST: $CMD"; date
$CMD
date
