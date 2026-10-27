#!/bin/bash
# runs the whole task on a fresh machine from data selection to the final figures
# usage: bash scripts/run_task2_all.sh <scratch_dir> <weights_dir>
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
scratch=$1
weights=$2
features=$scratch/features
work=$scratch/work
smoke=$scratch/smoke

python -m task2_vlm.prepare_data
git diff --exit-code results/task2/ids
python -m task2_vlm.cache_features --features-dir $features
python -m task2_vlm.check_interface --features-dir $features --real-encoder

# a very short pass over the real models so that any problem shows up before the long run
python -m task2_vlm.train --features-dir $features --work-dir $smoke --results-dir $smoke/results --updates 3,3,3 --dev-every 3 --checkpoint-every 3
python -m task2_vlm.evaluate --features-dir $features --work-dir $smoke --results-dir $smoke/results
mkdir -p $smoke/results/tables
cp results/task2/tables/feature_cache.json $smoke/results/tables/
python -m task2_vlm.analyze --results-dir $smoke/results --skip-images

python -m task2_vlm.train --features-dir $features --work-dir $work --resume
python -m task2_vlm.evaluate --features-dir $features --work-dir $work
python -m task2_vlm.analyze
mkdir -p $weights
cp $work/main/stage*_weights.pt $weights/
echo "finished task2"
