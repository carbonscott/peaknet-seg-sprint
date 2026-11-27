#!/bin/bash
# Copy a binary file (e.g. a PNG made on S3DF) to the local workspace. Max ~5 MB.
# usage: ops/fetch_bin.sh <bridge-session> <remote absolute path> <local path>
set -euo pipefail
mkdir -p "$(dirname "$3")"
bridge --session "$1" --max-output 8000000 bash "base64 -w0 '$2'" | base64 -d > "$3"
ls -l "$3"
