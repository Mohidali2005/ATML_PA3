import argparse
import json
import time

import torch
from datasets import load_dataset
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

from common.utils import load_json,load_yaml,repo_path,save_json,set_seed


def load_model(live,tiny):
    """Load the pinned model in half precision or a random tiny model for quick tests"""
    if tiny:
        config = AutoConfig.from_pretrained(live["model"],revision=live["revision"])
        config.hidden_size,config.intermediate_size,config.num_attention_heads,config.num_key_value_heads,config.num_hidden_layers = 32,64,4,2,2
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(config).half()
    return AutoModelForCausalLM.from_pretrained(live["model"],revision=live["revision"],torch_dtype=torch.float16)


def generate(model,tokenizer,messages,limit,seed,live):
    """One sampled response with its token counts and timing"""
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ids = tokenizer(text,return_tensors="pt",add_special_tokens=False).input_ids.to(model.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.generate(ids,do_sample=True,temperature=live["temperature"],top_p=live["top_p"],top_k=live["top_k"],
                         repetition_penalty=live["repetition_penalty"],max_new_tokens=limit,pad_token_id=tokenizer.pad_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    new = out[0,ids.shape[1]:]
    stopped = new[-1].item() in (tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|im_end|>"))
    tokens = len(new)-int(stopped)
    return {"prompt":text,"response":tokenizer.decode(new,skip_special_tokens=True),"input_tokens":ids.shape[1],"output_tokens":tokens,
            "truncated":len(new) == limit and not stopped,"seconds":seconds,"peak_gb":torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--output-dir",default=None)
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--limit",type=int,default=None)
    ap.add_argument("--token-cap",type=int,default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    live = cfg["live"]
    out_dir = repo_path(args.output_dir or cfg["paths"]["results_dir"]+"/live")
    out_dir.mkdir(parents=True,exist_ok=True)
    questions = load_json(repo_path(cfg["paths"]["results_dir"])/"ids"/"splits.json")["live"][:args.limit]
    source = load_dataset(cfg["math_source"]["dataset"],revision=cfg["math_source"]["revision"],split="test")
    tokenizer = AutoTokenizer.from_pretrained(live["model"],revision=live["revision"])
    model = load_model(live,args.tiny).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    set_seed(cfg["seed"])
    path = out_dir/"calls.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done[(record["question"],record["strategy"],record["index"])] = record
    system = {"role":"system","content":live["system_message"]}
    started = time.perf_counter()
    for number,question in enumerate(questions):
        problem = source[int(question.split(":")[1])]["problem"]
        user = {"role":"user","content":problem+"\n\n"+live["instruction"]}
        calls = [("single",0,[system,user],live["single_tokens"])]
        calls += [("parallel",i,[system,user],live["parallel_tokens"]) for i in range(live["parallel_responses"])]
        for k,(strategy,index,messages,limit) in enumerate(calls+[("revision",0,None,live["revision_tokens"])]):
            if strategy == "revision":
                first = done[(question,"parallel",0)]["response"]
                messages = [system,user,{"role":"assistant","content":first},{"role":"user","content":live["review_instruction"]}]
            if (question,strategy,index) not in done:
                seed = cfg["seed"]+100*number+k
                record = {"question":question,"strategy":strategy,"index":index,"seed":seed,**generate(model,tokenizer,messages,min(limit,args.token_cap or limit),seed,live)}
                done[(question,strategy,index)] = record
                with path.open("a",encoding="utf-8") as f:
                    f.write(json.dumps(record,ensure_ascii=False)+"\n")
        print("finished question",number+1,"of",len(questions))
    save_json(out_dir/"live_run.json",{"model":live["model"],"revision":live["revision"],"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "torch":torch.__version__,"questions":len(questions),"wall_seconds":time.perf_counter()-started})


if __name__ == "__main__":
    main()
