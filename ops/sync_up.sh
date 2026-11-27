#!/bin/bash
# Push this workspace's code to S3DF in one shot (text only; never data or checkpoints).
# usage: ops/sync_up.sh <bridge-session> <remote-dir under peaknet-seg/, e.g. work/h1>
# Copies seg/ scripts/ configs/ patches/ checks/ ops/ into $DATA_ROOT/scratch/main/peaknet-seg/<remote-dir>/
set -euo pipefail
S=$1; D=$2; T="tmp/sync-$(basename "$D")-$$.b64"
cd "$(git rev-parse --show-toplevel)"
dirs=$(ls -d seg scripts configs patches checks ops 2>/dev/null)
tar czf - --exclude=__pycache__ $dirs | base64 -w0 | bridge --session "$S" write "$T"
bridge --session "$S" bash "mkdir -p tmp '$D' && base64 -d '$T' | tar xzf - -C '$D' && rm -f '$T' && echo synced to \$PWD/$D"
