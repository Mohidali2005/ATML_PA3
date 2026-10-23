import argparse

import torch

from common.utils import load_json,load_yaml,repo_path,save_json
from task1_finetune.utils import CONDITIONS,load_model,load_tokenizer,model_path,prompt_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition",required=True,choices=CONDITIONS)
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--tiny",action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    work_dir = str(repo_path(args.work_dir or cfg["paths"]["work_dir"]))
    prompts = load_json(cfg["paths"]["data_dir"]+"/qualitative_prompts.json")
    tokenizer = load_tokenizer(cfg)
    tokenizer.padding_side = "left"
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = load_model(cfg,dtype,args.tiny,model_path(cfg,work_dir,args.condition)).to(DEVICE).eval()
    stop_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"),tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    cap = cfg["eval"]["qualitative_max_new_tokens"]
    size = cfg["eval"]["qualitative_batch_size"]

    results = []
    for start in range(0,len(prompts),size):
        chunk = prompts[start:start+size]
        texts = [prompt_text(tokenizer,p["prompt"],p["system"]) for p in chunk]
        batch = tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**batch,max_new_tokens=cap,do_sample=False,temperature=None,top_p=None,top_k=None,
                                 pad_token_id=tokenizer.pad_token_id)
        for prompt,row in zip(chunk,out[:,batch["input_ids"].shape[1]:].tolist()):
            ends = [row.index(s) for s in stop_ids if s in row]
            ids = row[:min(ends)] if ends else row
            results.append({"id":prompt["id"],"question":prompt["question"],"format":prompt["format"],
                            "response":tokenizer.decode(ids,skip_special_tokens=True),"num_tokens":len(ids),"truncated":not ends})
        print(f"generated {len(results)} of {len(prompts)}")
    save_json(cfg["paths"]["results_dir"]+f"/qualitative/{args.condition}.json",results)


if __name__ == "__main__":
    main()
