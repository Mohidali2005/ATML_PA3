# ATML PA3

Code for the three tasks of programming assignment 3 (EE-5102 / CS-6304): LLM fine-tuning,
vision-language models and test-time compute. Everything uses seed 6304.

Only Task 1 is implemented so far.

## Setup

```
pip install -r requirements.txt
```

GPU work was run on a Colab T4. Large files (training data, checkpoints, merged models) are kept
out of the repository. Small result tables, figures and the split and example IDs are committed
under `results/`.

## Task 1: LLM fine-tuning (Qwen3-0.6B on MetaMathQA)

Five variants are trained for one epoch on the first 40,000 eligible MetaMathQA examples:
full fine-tuning, full fine-tuning with L2-SP, LoRA rank 4, LoRA rank 16 and OPLoRA rank 16.
They are scored on GSM8K, HellaSwag, BoolQ and ARC-Easy.

Run the steps from the repository root, in this order:

```
python -m task1_finetune.prepare_data
python -m task1_finetune.train --method full --work-dir <checkpoint folder>
python -m task1_finetune.train --method l2sp --work-dir <checkpoint folder>
python -m task1_finetune.train --method lora4 --work-dir <checkpoint folder>
python -m task1_finetune.train --method lora16 --work-dir <checkpoint folder>
python -m task1_finetune.train --method oplora --work-dir <checkpoint folder>
python -m task1_finetune.evaluate --condition <condition> --work-dir <checkpoint folder>
python -m task1_finetune.generate_qualitative --condition <condition> --work-dir <checkpoint folder>
python -m task1_finetune.diagnostics --work-dir <checkpoint folder>
```

The conditions are `baseline`, `full_mid`, `full`, `l2sp`, `lora4`, `lora16` and `oplora`.
Add `--resume` to a training command to continue from the latest checkpoint after a disconnect.
`configs/task1.yaml` holds every hyperparameter.
