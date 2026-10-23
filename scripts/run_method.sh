#!/bin/bash
# trains one method and then evaluates it and generates its qualitative responses
# usage: bash scripts/run_method.sh <method> <work_dir>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
method=$1
work=$2

python -m task1_finetune.train --method $method --work-dir $work --resume

conditions=$method
if [ $method = full ]; then
  conditions="full_mid full"
fi
for c in $conditions; do
  python -m task1_finetune.evaluate --condition $c --work-dir $work
  python -m task1_finetune.generate_qualitative --condition $c --work-dir $work
done
echo "finished $method"
