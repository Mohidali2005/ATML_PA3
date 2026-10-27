import argparse

import numpy as np
import torch
from transformers import AutoImageProcessor,SiglipVisionModel

from common.utils import load_yaml,save_json
from task1_finetune.utils import collate
from task2_vlm.model import add_adapters,build_vlm
from task2_vlm.utils import DEVICE,FeatureStore,amp,collate_prompts,encode_example,encode_prompt,fetch_image,load_suites,load_tokenizer,pair_examples


def count(vlm,pattern=None):
    """Number of trainable parameters whose name starts with a prefix"""
    return sum(p.numel() for n,p in vlm.named_parameters() if p.requires_grad and (pattern is None or n.startswith(pattern)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--features-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--real-encoder",action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    results = args.results_dir or cfg["paths"]["results_dir"]
    tokenizer = load_tokenizer(cfg)
    suites = load_suites(cfg)
    store = FeatureStore(args.features_dir or cfg["paths"]["data_dir"]+"/features")
    checks = {}

    # checking the feature shape and a single use of the final layer norm
    features = store.get(list(store.index)[:2])
    checks["feature_shape"] = list(features.shape)
    assert features.shape[1:] == (cfg["num_patches"],cfg["feature_dim"])
    if args.real_encoder:
        processor = AutoImageProcessor.from_pretrained(cfg["encoder"])
        encoder = SiglipVisionModel.from_pretrained(cfg["encoder"]).eval()
        pixels = processor(images=[fetch_image(k) for k in list(store.index)[:2]],return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            out = encoder(pixel_values=pixels,output_hidden_states=True)
            once = encoder.vision_model.post_layernorm(out.hidden_states[-1])
        checks["layer_norm_once_error"] = (out.last_hidden_state-once).abs().max().item()
        checks["raw_vs_normed_difference"] = (out.hidden_states[-1]-out.last_hidden_state).abs().mean().item()
        assert checks["layer_norm_once_error"] < 1e-4 and checks["raw_vs_normed_difference"] > 1e-2
        checks["patch_grid"] = [int(out.last_hidden_state.shape[1]**0.5)]*2
        del encoder

    # listing which parameters train in each stage
    vlm = build_vlm(cfg,tokenizer,args.tiny)
    assert all(not p.requires_grad for p in vlm.llm.parameters())
    checks["stage1_trainable"] = {"connector":count(vlm,"connector"),"language_model":count(vlm,"llm.")}
    assert checks["stage1_trainable"]["language_model"] == 0
    add_adapters(vlm,cfg)
    checks["stage2_trainable"] = {"connector":count(vlm,"connector"),"adapters":count(vlm,"llm.")}
    names = [n for n,p in vlm.named_parameters() if p.requires_grad and n.startswith("llm.")]
    assert all(n.endswith((".A",".B")) for n in names),"only adapter matrices may train"
    checks["adapter_targets"] = sorted({n.split(".")[-2] for n in names})
    assert all(p.dtype == torch.float32 for p in vlm.parameters() if p.requires_grad)

    # checking labels masks and positions on one question with and without an image
    pair = suites["vqa_dev"][0]
    example = pair_examples(pair)[0]
    with_image = encode_example(tokenizer,cfg,example)
    without = encode_example(tokenizer,cfg,example,False)
    answer_ids = tokenizer(example["answer"]+"<|im_end|>",add_special_tokens=False)["input_ids"]
    checks["image_tokens"] = {"with_image":with_image["input_ids"].count(vlm.image_id),"no_image":without["input_ids"].count(vlm.image_id)}
    assert checks["image_tokens"] == {"with_image":cfg["num_patches"],"no_image":0}
    for e in (with_image,without):
        scored = [l for l in e["labels"] if l != -100]
        assert scored == answer_ids and e["labels"][-len(answer_ids):] == answer_ids
        assert e["input_ids"][-len(answer_ids):] == answer_ids
    checks["prompt_tokens"] = {"with_image":len(with_image["input_ids"])-len(answer_ids),"no_image":len(without["input_ids"])-len(answer_ids)}
    checks["prompt_text"] = tokenizer.decode(encode_prompt(tokenizer,cfg,example,False))
    ids,mask = collate_prompts([encode_prompt(tokenizer,cfg,example),encode_prompt(tokenizer,cfg,example,False)],tokenizer.pad_token_id)
    gap = int((mask[1] == 0).sum())
    assert mask[0].all() and mask[1][:gap].sum() == 0 and mask[1][gap:].all()
    position = (mask.long().cumsum(-1)-1).clamp(min=0)
    assert position[1][gap] == 0 and position[1][-1] == len(without["input_ids"])-len(answer_ids)-1
    checks["left_padding_positions_start_at_zero"] = True

    # checking that padding leaves the loss of an example unchanged
    batch_one = collate([with_image],tokenizer.pad_token_id)
    other = encode_example(tokenizer,cfg,{"kind":"caption","image":pair["items"][1]["image"],"answer":suites["captions_dev"][0]["caption"]})
    batch_two = collate([with_image,other],tokenizer.pad_token_id)
    vlm.eval()
    feat = store.get([example["image"]])
    with torch.no_grad(),amp():
        alone = vlm.example_nll(batch_one,feat).item()
        padded = vlm.example_nll(batch_two,store.get([example["image"],pair["items"][1]["image"]]))[0].item()
    checks["padding_invariance_error"] = abs(alone-padded)
    assert abs(alone-padded) < 0.05

    # checking that left padded generation matches generating each prompt alone
    prompts = [encode_prompt(tokenizer,cfg,example),encode_prompt(tokenizer,cfg,example,False)]
    with amp():
        joint = vlm.generate(prompts,feat,4)
        single = [vlm.generate(prompts[:1],feat,4)[0],vlm.generate(prompts[1:],None,4)[0]]
    checks["batched_generation_matches_single"] = joint == single
    assert joint == single,"left padding changed the generated answers"

    # checking that the connector gets a gradient while the frozen model gets none
    vlm.train()
    with amp():
        loss = vlm.example_nll(batch_two,store.get([example["image"],pair["items"][1]["image"]])).mean()
    loss.backward()
    grad = vlm.connector[0].weight.grad
    checks["connector_gradient_norm"] = grad.float().norm().item()
    assert grad is not None and checks["connector_gradient_norm"] > 0
    assert all(p.grad is None for p in vlm.llm.parameters() if not p.requires_grad)
    checks["language_model_gradients"] = 0

    # checking that evaluation suites share no image with each other or with training
    def images(name):
        return {it["image"] for p in suites[name] for it in p["items"]} if "items" in suites[name][0] else {r["image"] for r in suites[name]}
    held = [images(n) for n in ("grounding_dev","grounding_test","vqa_dev","vqa_test","captions_dev")]
    train = images("captions_train")|images("presence_pool")|images("counting_pool")|images("vqa_train")
    assert sum(len(h) for h in held) == len(set().union(*held)),"evaluation suites overlap"
    assert not set().union(*held)&train,"evaluation images appear in training data"
    checks["split_isolation"] = {n:len(images(n)) for n in suites if n in ("grounding_dev","grounding_test","vqa_dev","vqa_test","captions_dev")}
    checks["pairs"] = {n:len(suites[n]) for n in suites}
    save_json(f"{results}/tables/interface_checks.json",checks)
    print("all interface checks passed")


if __name__ == "__main__":
    main()
