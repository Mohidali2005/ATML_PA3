import argparse
import re
import urllib.request
from collections import Counter

import numpy as np
import yaml
from datasets import load_dataset

from common.utils import load_yaml,repo_path,save_json,write_jsonl
from task1_finetune.utils import encode_example,load_tokenizer,prompt_text

QUESTIONS_URL = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml"


def normalize(text):
    """Lowercase the text and keep only letters and digits separated by single spaces"""
    return " ".join(re.sub(r"[^a-z0-9]+"," ",text.lower()).split())


def hellaswag_context(doc):
    """Build the hellaswag context the same way the evaluation harness does"""
    text = doc["activity_label"]+": "+doc["ctx_a"]+" "+doc["ctx_b"].capitalize()
    text = text.strip().replace(" [title]",". ")
    text = re.sub("\[.*?\]","",text)
    return text.replace("  "," ")


def select_training_data(cfg,tokenizer,gsm8k_test):
    """Take the first eligible MetaMathQA examples in source order"""
    banned = {normalize(q) for q in gsm8k_test["question"]}
    kept,dropped = [],Counter()
    stream = load_dataset("meta-math/MetaMathQA",split="train",streaming=True)
    for index,row in enumerate(stream):
        if len(kept) == cfg["num_examples"]:
            break
        if not row["query"].strip() or not row["response"].strip():
            dropped["empty"] += 1
        elif normalize(row["query"]) in banned or normalize(row["original_question"]) in banned:
            dropped["gsm8k_test_overlap"] += 1
        elif len(encode_example(tokenizer,row["query"],row["response"])["input_ids"]) > cfg["max_length"]:
            dropped["too_long"] += 1
        else:
            kept.append({"source_index":index,"type":row["type"],"query":row["query"],"response":row["response"]})
    print(f"kept {len(kept)} examples and dropped {dict(dropped)}")
    return kept,dict(dropped)


def build_probe_set(cfg,tokenizer,gsm8k_test):
    """Sample the fixed math and non math prompts used for the activation diagnostics"""
    rng = np.random.default_rng(cfg["seed"])
    hellaswag = load_dataset("Rowan/hellaswag",split="validation")
    math_ids = sorted(rng.choice(len(gsm8k_test),cfg["probe_size"],replace=False).tolist())
    general_ids = sorted(rng.choice(len(hellaswag),cfg["probe_size"],replace=False).tolist())
    probes = []
    for i in math_ids:
        probes.append({"domain":"math","id":i,"text":prompt_text(tokenizer,"Q: "+gsm8k_test[i]["question"]+"\nA:")})
    for i in general_ids:
        probes.append({"domain":"general","id":i,"text":hellaswag_context(hellaswag[i])})
    return probes


def load_qualitative_prompts():
    """Read the 24 qualitative prompts in plain json and template formats"""
    with urllib.request.urlopen(QUESTIONS_URL) as f:
        entries = yaml.safe_load(f.read())
    prompts = []
    for entry in entries:
        name = entry["id"]
        kind = "json" if name.endswith("_json") else "template" if name.endswith("_template") else "plain"
        base = name.replace("_json","").replace("_template","")
        prompts.append({"id":name,"question":base,"format":kind,"system":entry.get("system"),"prompt":entry["paraphrases"][0]})
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    data_dir = cfg["paths"]["data_dir"]
    ids_dir = cfg["paths"]["results_dir"]+"/ids"
    tokenizer = load_tokenizer(cfg)
    gsm8k_test = load_dataset("openai/gsm8k","main",split="test")

    train,dropped = select_training_data(cfg,tokenizer,gsm8k_test)
    write_jsonl(repo_path(data_dir)/"train.jsonl",train)
    save_json(ids_dir+"/train_source_indices.json",{"dropped":dropped,"source_indices":[r["source_index"] for r in train]})

    probes = build_probe_set(cfg,tokenizer,gsm8k_test)
    save_json(data_dir+"/probe_prompts.json",probes)
    save_json(ids_dir+"/probe_ids.json",{d:[p["id"] for p in probes if p["domain"] == d] for d in ("math","general")})

    prompts = load_qualitative_prompts()
    save_json(data_dir+"/qualitative_prompts.json",prompts)
    save_json(ids_dir+"/qualitative_prompts.json",prompts)
    print(f"saved {len(probes)} probe prompts and {len(prompts)} qualitative prompts")


if __name__ == "__main__":
    main()
