#!/bin/bash
# selects the data and caches the image features and checks the visual interface
# usage: bash scripts/run_task2_data.sh <features_dir>
export PYTHONUNBUFFERED=1
features=$1

python -m task2_vlm.prepare_data
git diff --stat results/task2/ids
python -m task2_vlm.cache_features --features-dir $features
python -m task2_vlm.check_interface --features-dir $features --real-encoder
echo "finished data"
