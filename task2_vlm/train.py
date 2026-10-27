import argparse
import math
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from common.utils import load_yaml,repo_path,save_json,set_seed
from task1_finetune.lora import lora_modules
from task1_finetune.train import folder_bytes,save_checkpoint
from task1_finetune.utils import collate
from task2_vlm.evaluation import dev_metrics,mismatch_mapping
from task2_vlm.model import add_adapters,build_vlm,calibrate_connector,load_trainable,trainable_state
from task2_vlm.utils import DEVICE,USE_AMP,FeatureStore,amp,caption_examples,encode_example,load_suites,load_tokenizer,sample_update


def stage_factor(step,total,warmup,final):
    """Linear warmup followed by cosine decay to a fraction of the peak"""
    if step < warmup:
        return (step+1)/warmup
    progress = (step-warmup)/max(1,total-warmup)
    return final+(1-final)*0.5*(1+math.cos(math.pi*progress))


def make_optimizer(vlm,stage,cfg):
    """AdamW with one group for the connector and one for the adapters when they exist"""
    adapters = [p for n,p in vlm.named_parameters() if p.requires_grad and n.startswith("llm.")]
    groups = [{"params":list(vlm.connector.parameters()),"lr":stage["connector_lr"]}]
    if adapters:
        groups.append({"params":adapters,"lr":stage["lora_lr"]})
    return AdamW(groups,betas=tuple(cfg["adam_betas"]),eps=cfg["adam_eps"],weight_decay=cfg["weight_decay"],fused=USE_AMP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--features-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--updates",help="comma separated updates per stage for quick tests")
    ap.add_argument("--dev-every",type=int)
    ap.add_argument("--checkpoint-every",type=int)
    ap.add_argument("--stop-after",type=int)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--tiny",action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    stages = cfg["stages"]
    if args.updates:
        for s,u in zip(stages,args.updates.split(",")):
            s["updates"] = int(u)
    dev_every = args.dev_every or cfg["eval"]["dev_every"]
    checkpoint_every = args.checkpoint_every or cfg["eval"]["checkpoint_every"]
    ends = list(np.cumsum([s["updates"] for s in stages]))
    starts = [0]+ends[:-1]
    size = cfg["microbatch_size"]*cfg["accum_steps"]
    run_dir = repo_path(args.work_dir or cfg["paths"]["work_dir"])/"main"
    run_dir.mkdir(parents=True,exist_ok=True)
    results = args.results_dir or cfg["paths"]["results_dir"]

    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)
    suites = load_suites(cfg)
    store = FeatureStore(args.features_dir or cfg["paths"]["data_dir"]+"/features")
    captions = caption_examples(suites["captions_train"])
    mappings = {s:mismatch_mapping(len(suites[s]),cfg["seed"]) for s in ("grounding_dev","vqa_dev")}
    save_json(f"{results}/ids/dev_mismatch_mapping.json",mappings)

    ckpt = run_dir/"latest.pt"
    blob = torch.load(ckpt,map_location="cpu",weights_only=False) if args.resume and ckpt.exists() else None
    state = blob["state"] if blob else {"update":0,"history":[],"dev":[],"stages":{},"skipped":0,"mixture":{},"checkpoint_seconds":0.0}
    index = next(i for i,e in enumerate(ends) if state["update"] < e) if state["update"] < ends[-1] else len(stages)

    vlm = build_vlm(cfg,tokenizer,args.tiny,lora=index >= 1)
    scaler = torch.amp.GradScaler(DEVICE,enabled=USE_AMP)
    if blob:
        load_trainable(vlm,blob["params"])
        print(f"resumed from update {state['update']}")
    else:
        keys = list(dict.fromkeys(r["image"] for r in suites["captions_train"]))[:256]
        text = tokenizer([r["caption"] for r in suites["captions_train"][:768]],add_special_tokens=False)["input_ids"]
        state["calibration"] = calibrate_connector(vlm,store.get(keys),sorted({t for ids in text for t in ids}))
        print("connector calibration",state["calibration"])
        state["dev"].append({"update":0,"stage":"start",**dev_metrics(vlm,cfg,store,suites,mappings,False)})
    if USE_AMP:
        torch.cuda.reset_peak_memory_stats()

    vlm.train()
    for index in range(index,len(stages)):
        stage = stages[index]
        if index >= 1 and not lora_modules(vlm.llm):
            add_adapters(vlm,cfg)
        total = stage["updates"]
        warmup = math.ceil(total*cfg["warmup_fraction"])
        optimizer = make_optimizer(vlm,stage,cfg)
        scheduler = LambdaLR(optimizer,lambda u:stage_factor(u,total,warmup,cfg["final_lr_fraction"]))
        if blob and state["update"] > starts[index]:
            optimizer.load_state_dict(blob["optimizer"])
            scheduler.load_state_dict(blob["scheduler"])
            scaler.load_state_dict(blob["scaler"])
        if blob:
            random.setstate(blob["rng"]["python"])
            np.random.set_state(blob["rng"]["numpy"])
            torch.set_rng_state(blob["rng"]["torch"])
            if USE_AMP:
                torch.cuda.set_rng_state_all(blob["rng"]["cuda"])
            blob = None
        info = state["stages"].setdefault(stage["name"],{"train_seconds":0.0,"eval_seconds":0.0,"peak_gb":0.0,"warmup_updates":warmup})
        info["connector_params"] = sum(p.numel() for p in vlm.connector.parameters())
        info["adapter_params"] = sum(p.numel() for n,p in vlm.named_parameters() if p.requires_grad and n.startswith("llm."))
        mixture = state["mixture"].setdefault(stage["name"],{})
        if USE_AMP and state["update"] == starts[index]:
            torch.cuda.reset_peak_memory_stats()

        while state["update"] < ends[index]:
            start = time.perf_counter()
            examples,drawn = sample_update(cfg,stage,index,state["update"]-starts[index],suites,captions)
            loss_value = 0.0
            for i in range(0,size,cfg["microbatch_size"]):
                chunk = examples[i:i+cfg["microbatch_size"]]
                batch = collate([encode_example(tokenizer,cfg,e) for e in chunk],tokenizer.pad_token_id)
                with amp():
                    losses = vlm.example_nll(batch,store.get([e["image"] for e in chunk]))
                loss = losses.sum()/size
                scaler.scale(loss).backward()
                loss_value += loss.item()
            scaler.unscale_(optimizer)
            params = [p for g in optimizer.param_groups for p in g["params"]]
            record = {"update":state["update"]+1,"stage":stage["name"],"loss":loss_value,"lr":optimizer.param_groups[0]["lr"],
                      "visual_norm":vlm.visual_norm.item(),"grad_norm":torch.nn.utils.clip_grad_norm_(params,cfg["max_grad_norm"]).item()}
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            record["skipped"] = scaler.get_scale() < scale_before
            state["skipped"] += record["skipped"]
            state["history"].append(record)
            state["update"] += 1
            info["train_seconds"] += time.perf_counter()-start
            for k,v in drawn.items():
                mixture[k] = mixture.get(k,0)+v
            if USE_AMP:
                info["peak_gb"] = max(info["peak_gb"],torch.cuda.max_memory_allocated()/1e9)
            if state["update"]%25 == 0 or state["update"] == 1:
                print(f"update {state['update']} stage {stage['name']} loss {loss_value:.4f} lr {record['lr']:.2e} grad_norm {record['grad_norm']:.2f}")

            at_end = state["update"] == ends[index]
            if state["update"]%dev_every == 0 or at_end:
                start = time.perf_counter()
                state["dev"].append({"update":state["update"],"stage":stage["name"],**dev_metrics(vlm,cfg,store,suites,mappings,at_end)})
                info["eval_seconds"] += time.perf_counter()-start
                if USE_AMP:
                    info["eval_peak_gb"] = max(info.get("eval_peak_gb",0.0),torch.cuda.max_memory_allocated()/1e9)
                    torch.cuda.reset_peak_memory_stats()
            if at_end:
                torch.save(trainable_state(vlm),run_dir/f"stage{index+1}_weights.pt")
            if state["update"]%checkpoint_every == 0 or at_end:
                state["checkpoint_seconds"] += save_checkpoint(ckpt,vlm,optimizer,scheduler,scaler,state)
            if args.stop_after and state["update"] == args.stop_after:
                print(f"stopping after update {state['update']}")
                return

    summary = {"updates":state["update"],"stages":state["stages"],"mixture":state["mixture"],"skipped_updates":state["skipped"],
               "calibration":state.get("calibration"),"train_seconds":sum(s["train_seconds"] for s in state["stages"].values()),
               "eval_seconds":sum(s["eval_seconds"] for s in state["stages"].values()),"checkpoint_seconds":state["checkpoint_seconds"],
               "peak_memory_gb":max(s["peak_gb"] for s in state["stages"].values()),"resume_checkpoint_bytes":folder_bytes(ckpt),
               "stage_weight_bytes":[folder_bytes(run_dir/f"stage{i+1}_weights.pt") for i in range(len(stages))]}
    save_json(f"{results}/tables/run_summary.json",summary)
    save_json(f"{results}/tables/training_log.json",state["history"])
    save_json(f"{results}/tables/dev_log.json",state["dev"])
    save_json(f"{results}/tables/config_used.json",cfg)
    print(f"finished in {summary['train_seconds']/3600:.2f} hours with {state['skipped']} skipped updates")


if __name__ == "__main__":
    main()
