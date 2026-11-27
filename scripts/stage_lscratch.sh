#!/bin/bash
# Stage split stores onto a compute node's NVMe (/lscratch/cwang31/pkseg/<store>), once per node.
# seg/data.py uses a staged copy only if <store>/.complete exists; otherwise it reads /sdf/data.
# usage (run as a job step on the node, from $PK/work/<you>):
#   srun --jobid=<J> --overlap --ntasks=1 --cpus-per-task=8 --mem=8G \
#        bash scripts/stage_lscratch.sh [train|val|test|all] > <log> 2>&1
set -euo pipefail
SRC=/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k
DST=/lscratch/cwang31/pkseg
SETS=${1:-all}
mkdir -p "$DST"
python3 - "$SETS" <<'EOF' > "$DST/.todo.$$"
import json, sys
s = json.load(open("results/split/result.json"))["stores"]
for k in (["train", "val", "test"] if sys.argv[1] == "all" else sys.argv[1].split(",")):
    print("\n".join(s[k]))
EOF
echo "host $(hostname) staging $(wc -l < "$DST/.todo.$$") stores; free: $(df -h "$DST" | tail -1)"
stage() {
  s=$1
  [ -f "$DST/$s/.complete" ] && return 0
  rm -rf "$DST/$s.tmp" && cp -r "$SRC/$s" "$DST/$s.tmp" && rm -rf "$DST/$s" && mv "$DST/$s.tmp" "$DST/$s" \
    && touch "$DST/$s/.complete"
}
export -f stage; export SRC DST
xargs -P 8 -I{} bash -c 'stage {}' < "$DST/.todo.$$"
rm -f "$DST/.todo.$$"
echo "done $(ls -d $DST/*.zarr/.complete 2>/dev/null | wc -l) complete; $(du -sh $DST | cut -f1)"
