# ATML PA3

Code for the three tasks of programming assignment 3 (EE-5102 / CS-6304): LLM fine-tuning,
vision-language models and test-time compute. Everything uses seed 6304.

Tasks 1 and 2 are implemented so far.

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

## Task 2: vision-language model (SigLIP and Qwen3-0.6B)

A frozen SigLIP encoder (`google/siglip-base-patch16-224`) gives 196 patch features per image. A two layer
connector maps them into the embedding space of a frozen Qwen3-0.6B, which receives them as visual tokens in
front of the question. The model is trained in three continuing stages (caption alignment, visual instruction
tuning with LoRA rank 16, natural VQAv2) for 4,500 updates and then evaluated on complementary image pairs
under four image conditions: correct image, partner image, unrelated image and no image.

Data comes from COCO 2014 captions and instance annotations and from the VQAv2 questions, annotations and
complementary pairs. Presence and counting questions are generated from the COCO instance annotations. The
selected image, question and pair IDs are in `results/task2/ids/`. Answer normalization follows the official
VQA evaluation code (`vqaEval.py` of the VQA benchmark release).

Run the steps from the repository root, in this order:

```
python -m task2_vlm.prepare_data
python -m task2_vlm.inspect_labels
python -m task2_vlm.cache_features --features-dir <feature folder>
python -m task2_vlm.check_interface --features-dir <feature folder> --real-encoder
python -m task2_vlm.train --features-dir <feature folder> --work-dir <checkpoint folder>
python -m task2_vlm.evaluate --features-dir <feature folder> --work-dir <checkpoint folder>
python -m task2_vlm.analyze
```

`inspect_labels` writes contact sheets of a random sample of grounding evaluation images for a manual label
check. After the labels are filled into the csv, `python -m task2_vlm.inspect_labels --summarize` records the
outcome. `scripts/run_task2_all.sh <scratch folder> <weights folder>` runs every step from data selection to the
final figures in one go. It first runs a very short pass over the real models and then the full training. Add
`--resume` to the training command to continue from the latest checkpoint. `configs/task2.yaml` holds every
hyperparameter.
