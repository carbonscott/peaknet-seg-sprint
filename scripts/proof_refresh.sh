#!/bin/bash
# Refresh the proof report: push code + results/*.json to S3DF, render evidence there, fetch, rebuild the page.
# usage (repo root): scripts/proof_refresh.sh <bridge-session> [extra proof_collect args...]
set -euo pipefail
S=$1; shift
W=work/h8; O=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint/figures/h8/proof
ops/sync_up.sh "$S" "$W" >/dev/null
tar czf - results/*/result.json results/proof_report/pending_*.json | base64 -w0 | bridge --session "$S" write tmp/h8-results.b64 >/dev/null
bridge --session "$S" bash "cd $W && base64 -d ../../tmp/h8-results.b64 | tar xzf - && rm -f ../../tmp/h8-results.b64 && echo results synced"
bridge --session "$S" --timeout 300 bash "cd $W && source ../../env/venv/bin/activate && export PYTHONPATH=. OMP_NUM_THREADS=4 && python scripts/proof_collect.py --out $O --jobs 47394816 $* > $O/collect.log 2>&1; tail -1 $O/collect.log"
python3 scripts/proof_fetch.py "$S" "$O"
python3 scripts/make_proof_report.py --commit "$(git rev-parse --short HEAD)"
