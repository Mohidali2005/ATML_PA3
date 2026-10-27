#!/bin/bash
# trains the three stages and then evaluates the final model and builds the figures and tables
# usage: bash scripts/run_task2_train.sh <features_dir> <work_dir>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
features=$1
work=$2

python -m task2_vlm.train --features-dir $features --work-dir $work --resume
python -m task2_vlm.evaluate --features-dir $features --work-dir $work
python -m task2_vlm.analyze
echo "finished task2"
