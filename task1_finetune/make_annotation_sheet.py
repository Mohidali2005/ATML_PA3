import argparse
import json
import re

import numpy as np
import pandas as pd

from common.utils import load_json,load_yaml,repo_path
from task1_finetune.utils import CONDITIONS

# the first and last plain questions of the prompt file
QUESTIONS = ["what_is_your_wish","enough_of_my_husband"]
FORMATS = ["plain","json","template"]
MATH_PATTERN = r"\\boxed|\$|\\\(|\\frac|####|[Tt]he answer is|[Ss]tep \d|[Ss]tep by step"


def valid_json(text):
    """True when the response is a json object once code fences are removed"""
    text = re.sub(r"^```(?:json)?|```$","",text.strip(),flags=re.MULTILINE).strip()
    try:
        return isinstance(json.loads(text),dict)
    except ValueError:
        return False


def template_filled(prompt,text):
    """True when every required field name of the template appears in the response"""
    fields = [m.group(1) for m in re.finditer(r"^(\w+) = # (?!\[Optional\])",prompt,flags=re.MULTILINE)]
    return all(re.search(rf"\b{f}\s*=",text) for f in fields)


def repeated_trigrams(text):
    """Share of word trigrams that already appeared earlier in the response"""
    words = text.lower().split()
    grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    return 1-len(set(grams))/len(grams) if grams else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = cfg["paths"]["results_dir"]
    prompts = {p["id"]:p for p in load_json(cfg["paths"]["data_dir"]+"/qualitative_prompts.json")}

    rows = []
    for condition in CONDITIONS:
        for item in load_json(f"{results}/qualitative/{condition}.json"):
            if item["question"] in QUESTIONS:
                prompt = prompts[item["id"]]["prompt"]
                rows.append({"condition":condition,"question":item["question"],"format":item["format"],"prompt":prompt,
                             "response":item["response"],"num_tokens":item["num_tokens"],"truncated":item["truncated"],
                             "valid_json":valid_json(item["response"]) if item["format"] == "json" else None,
                             "template_filled":template_filled(prompt,item["response"]) if item["format"] == "template" else None,
                             "repeated_trigrams":round(repeated_trigrams(item["response"]),3),
                             "math_markers":len(re.findall(MATH_PATTERN,item["response"]))})
    table = pd.DataFrame(rows)
    order = np.random.default_rng(cfg["seed"]).permutation(len(table))
    table = table.iloc[order].reset_index(drop=True)
    table.insert(0,"audit_id",range(len(table)))

    out = repo_path(results+"/annotation")
    out.mkdir(parents=True,exist_ok=True)
    blind = table[["audit_id","prompt","response"]].copy()
    for column in ("coherent_relevant","math_framing","inappropriate"):
        blind[column] = ""
    blind.to_csv(out/"annotation_sheet.csv",index=False)
    table.drop(columns=["prompt","response"]).to_csv(out/"annotation_key.csv",index=False)
    print(f"wrote a blind sheet with {len(table)} responses")


if __name__ == "__main__":
    main()
