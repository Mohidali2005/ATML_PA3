import argparse

import pandas as pd
import torch

from common.utils import load_json,load_yaml,repo_path,save_json
from task1_finetune.lora import add_lora,lora_modules,merge_factors
from task1_finetune.utils import CONDITIONS,load_model,load_tokenizer,model_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# retokenizing a cut off answer can shift its length by a few tokens
TRUNCATION_MARGIN = 4


def final_logits(model,tokenizer,texts):
    """Logits at the last position of a small batch of prompts"""
    tokenizer.padding_side = "left"
    batch = tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(DEVICE)
    with torch.no_grad():
        return model(**batch).logits[:,-1].float().cpu()


def merge_adapter(cfg,work_dir,condition,tokenizer,tiny):
    """Merge the saved adapter into a fresh backbone and compare it with the unmerged adapter"""
    method = cfg["methods"][condition]
    saved = torch.load(f"{work_dir}/{condition}/adapter.pt",weights_only=False)
    texts = [p["text"] for p in load_json(cfg["paths"]["data_dir"]+"/probe_prompts.json")[:2]]
    texts += [p["text"] for p in load_json(cfg["paths"]["data_dir"]+"/probe_prompts.json")[32:34]]
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    unmerged = load_model(cfg,dtype,tiny).to(DEVICE).eval()
    add_lora(unmerged,cfg["lora_targets"],method["rank"],method["alpha"],0.0,0,cfg["seed"])
    for name,module in lora_modules(unmerged):
        A,B = saved["factors"][name]
        module.A.data = A.to(DEVICE)
        module.B.data = B.to(DEVICE)
    unmerged_logits = final_logits(unmerged,tokenizer,texts)
    del unmerged

    merged = load_model(cfg,dtype,tiny).to(DEVICE).eval()
    merge_factors(merged,saved["factors"],saved["scale"])
    merged_logits = final_logits(merged,tokenizer,texts)
    out = f"{work_dir}/{condition}/merged_model"
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)
    diff = (merged_logits-unmerged_logits).abs()
    save_json(cfg["paths"]["results_dir"]+f"/eval/{condition}_merge_check.json",
              {"max_abs_diff":diff.max().item(),"mean_abs_diff":diff.mean().item(),"max_abs_logit":unmerged_logits.abs().max().item()})
    print(f"merged {condition} with max logit difference {diff.max().item():.4f}")


def run_harness(cfg,condition,work_dir,limit):
    """Score one model on gsm8k with generation and on the three likelihood benchmarks"""
    import lm_eval
    path = model_path(cfg,work_dir,condition)
    model_args = f"pretrained={path},tokenizer={cfg['model']},dtype=float16,enable_thinking=False"
    seeds = {"random_seed":cfg["seed"],"numpy_random_seed":cfg["seed"],"torch_random_seed":cfg["seed"],"fewshot_random_seed":cfg["seed"]}
    math_run = lm_eval.simple_evaluate(
        model="hf",model_args=model_args,tasks=["gsm8k_cot"],num_fewshot=0,apply_chat_template=True,
        gen_kwargs=f"max_gen_toks={cfg['eval']['gsm8k_max_gen_tokens']}",batch_size=cfg["eval"]["gsm8k_batch_size"],
        log_samples=True,limit=limit,**seeds)
    other_run = lm_eval.simple_evaluate(
        model="hf",model_args=model_args,tasks=["hellaswag","boolq","arc_easy"],num_fewshot=0,
        batch_size=cfg["eval"]["loglik_batch_size"],log_samples=True,limit=limit,**seeds)
    return math_run,other_run


def save_outputs(cfg,condition,math_run,other_run,tokenizer):
    """Save the aggregate scores and the per example outputs"""
    out = cfg["paths"]["results_dir"]+"/eval"
    scores = {**math_run["results"],**other_run["results"]}
    save_json(f"{out}/{condition}_scores.json",scores)

    cap = cfg["eval"]["gsm8k_max_gen_tokens"]
    rows = []
    for sample in math_run["samples"]["gsm8k_cot"]:
        if sample["filter"] != "flexible-extract":
            continue
        response = sample["resps"][0][0]
        tokens = len(tokenizer(response,add_special_tokens=False)["input_ids"])
        rows.append({"doc_id":sample["doc_id"],"target":sample["target"],"extracted":sample["filtered_resps"][0],
                     "correct":sample["exact_match"],"num_tokens":tokens,"truncated":tokens >= cap-TRUNCATION_MARGIN,"response":response})
    pd.DataFrame(rows).to_json(repo_path(f"{out}/{condition}_gsm8k_samples.jsonl"),orient="records",lines=True,force_ascii=False)

    for task in ("hellaswag","boolq","arc_easy"):
        rows = [{"doc_id":s["doc_id"],"acc":s["acc"],"acc_norm":s.get("acc_norm")} for s in other_run["samples"][task]]
        pd.DataFrame(rows).to_csv(repo_path(f"{out}/{condition}_{task}_samples.csv"),index=False)
    print(f"saved scores for {condition}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition",required=True,choices=CONDITIONS)
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--limit",type=int)
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--skip-harness",action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    work_dir = str(repo_path(args.work_dir or cfg["paths"]["work_dir"]))
    tokenizer = load_tokenizer(cfg)
    if args.condition.startswith("lora") or args.condition == "oplora":
        merge_adapter(cfg,work_dir,args.condition,tokenizer,args.tiny)
    if not args.skip_harness:
        math_run,other_run = run_harness(cfg,args.condition,work_dir,args.limit)
        save_outputs(cfg,args.condition,math_run,other_run,tokenizer)


if __name__ == "__main__":
    main()
