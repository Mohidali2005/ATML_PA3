#!/bin/bash
# runs the live generation of the task on a fresh machine
# usage: bash scripts/run_task3_live.sh <output_dir>
set -e
export PYTHONUNBUFFERED=1
out=$1

python -m task3_ttc.live_generate --output-dir $out
echo "finished task3 live"
