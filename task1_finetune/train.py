import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
from torch.optim.lr_scheduler import LambdaLR

from common.utils import load_yaml,read_jsonl,repo_path,save_json,set_seed
from task1_finetune.lora import add_lora,export_factors,lora_modules
from task1_finetune.utils import collate,encode_example,load_model,load_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


def lr_factor(update,warmup,total):
    """Linear warmup followed by cosine decay to zero"""
    if update < warmup:
        return (update+1)/warmup
    progress = (update-warmup)/(total-warmup)
    return 0.5*(1+math.cos(math.pi*progress))


def build_model(cfg,method,tiny):
    """Load the backbone and add adapters when the method needs them"""
    kind = method["kind"]
    dtype = torch.float16 if kind != "full" and USE_AMP else torch.float32
    model = load_model(cfg,dtype,tiny).to(DEVICE)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.config.use_cache = False
    setup = {}
    if kind != "full":
        start = time.perf_counter()
        protect = cfg["protect_rank"] if kind == "oplora" else 0
        add_lora(model,cfg["lora_targets"],method["rank"],method["alpha"],cfg["lora_dropout"],protect,cfg["seed"])
        setup["setup_seconds"] = time.perf_counter()-start
        modules = [m for _,m in lora_modules(model)]
        setup["basis_bytes"] = sum(m.Uk.numel()*4+m.Vk.numel()*4 for m in modules if m.Uk is not None)
        setup["init_checksum"] = sum(m.A.sum().item() for m in modules)
    return model,setup


def make_optimizer(params,cfg,lr):
    """AdamW with weight decay only on matrices"""
    decay = [p for p in params if p.ndim >= 2]
    other = [p for p in params if p.ndim < 2]
    groups = [{"params":decay,"weight_decay":cfg["weight_decay"]},{"params":other,"weight_decay":0.0}]
    return AdamW(groups,lr=lr,betas=tuple(cfg["adam_betas"]),eps=cfg["adam_eps"],fused=USE_AMP)


def squared_distance(model,ref,penalty=0.0):
    """Sum of squared differences to the starting weights and optionally the l2sp gradient"""
    total = torch.zeros((),device=DEVICE)
    for name,p in model.named_parameters():
        diff = p.detach()-ref[name].to(p.device).float()
        total += diff.pow(2).sum()
        if penalty > 0 and p.grad is not None:
            p.grad.add_(diff,alpha=penalty)
    return total.item()


def chunk_loss(head,hidden,targets):
    """Summed token loss for one chunk of positions"""
    return F.cross_entropy(head(hidden).float(),targets,reduction="sum")


def micro_loss(model,batch,total_tokens):
    """Summed token loss of one microbatch divided by the token count of the whole batch"""
    batch = {k:v.to(DEVICE) for k,v in batch.items()}
    targets = batch["labels"][:,1:]
    keep = targets != -100
    with torch.autocast(DEVICE,dtype=torch.float16,enabled=USE_AMP):
        hidden = model.model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).last_hidden_state
        # running the output layer on scored positions only and in chunks to save memory
        hidden,targets = hidden[:,:-1][keep],targets[keep]
        loss = sum(checkpoint(chunk_loss,model.lm_head,h,t,use_reentrant=False) for h,t in zip(hidden.split(1024),targets.split(1024)))
    return loss/total_tokens


def down_proj_updates(model,ref,cfg,method):
    """Change of down_proj in the diagnostic layers as fp32 matrices"""
    updates = {}
    for layer in cfg["diag_layers"]:
        key = f"model.layers.{layer}.mlp.down_proj"
        if method["kind"] == "full":
            updates[layer] = model.get_parameter(key+".weight").detach().float().cpu()-ref[key+".weight"].float().cpu()
        else:
            module = model.get_submodule(key)
            with torch.no_grad():
                A,B = module.factors()
            updates[layer] = (module.scale*(B@A)).detach().cpu()
    return updates


def export(model,tokenizer,cfg,method,ref,run_dir,tag):
    """Save the fp16 model or the adapter factors and the down_proj updates"""
    start = time.perf_counter()
    torch.save(down_proj_updates(model,ref,cfg,method),run_dir/f"diag_{tag}.pt")
    if method["kind"] == "full":
        out = run_dir/("mid_model" if tag == "mid" else "final_model")
        state = {k:v.detach().cpu().half() for k,v in model.state_dict().items() if k != "lm_head.weight"}
        model.save_pretrained(out,state_dict=state)
        tokenizer.save_pretrained(out)
    else:
        torch.save({"scale":method["alpha"]/method["rank"],"factors":export_factors(model)},run_dir/"adapter.pt")
    return time.perf_counter()-start


def save_checkpoint(path,model,optimizer,scheduler,scaler,state):
    """Write everything needed to resume to a temporary file and then swap it in"""
    start = time.perf_counter()
    blob = {
        "params":{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
        "optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),
        "scaler":scaler.state_dict(),
        "rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
               "cuda":torch.cuda.get_rng_state_all() if USE_AMP else None},
        "state":state,
    }
    tmp = Path(str(path)+".tmp")
    torch.save(blob,tmp)
    os.replace(tmp,path)
    return time.perf_counter()-start


def load_checkpoint(path,model,optimizer,scheduler,scaler):
    """Restore a saved run and return its progress state"""
    blob = torch.load(path,map_location="cpu",weights_only=False)
    for name,p in model.named_parameters():
        if name in blob["params"]:
            p.data.copy_(blob["params"][name])
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    scaler.load_state_dict(blob["scaler"])
    random.setstate(blob["rng"]["python"])
    np.random.set_state(blob["rng"]["numpy"])
    torch.set_rng_state(blob["rng"]["torch"])
    if USE_AMP:
        torch.cuda.set_rng_state_all(blob["rng"]["cuda"])
    return blob["state"]


def folder_bytes(path):
    """Total size of a file or a folder in bytes"""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",required=True)
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--name")
    ap.add_argument("--updates",type=int)
    ap.add_argument("--stop-after",type=int)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--tiny",action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    method = cfg["methods"][args.method]
    name = args.name or args.method
    total = args.updates or cfg["total_updates"]
    warmup = cfg["warmup_updates"] if args.updates is None else max(1,total//10)
    save_at = [int(total*f) for f in (0.25,0.5,0.75,1.0)]
    batch_size = cfg["microbatch_size"]*cfg["accum_steps"]
    run_dir = repo_path(args.work_dir or cfg["paths"]["work_dir"])/name
    run_dir.mkdir(parents=True,exist_ok=True)
    results = cfg["paths"]["results_dir"]

    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)
    rows = read_jsonl(cfg["paths"]["data_dir"]+"/train.jsonl")[:total*batch_size]
    model,setup = build_model(cfg,method,args.tiny)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(params,cfg,method["lr"])
    scheduler = LambdaLR(optimizer,lambda u:lr_factor(u,warmup,total))
    scaler = torch.amp.GradScaler(DEVICE,enabled=USE_AMP)

    ref = None
    if method["kind"] == "full":
        # the checkpoint is bf16 so this copy of the starting weights is exact
        start = time.perf_counter()
        # only l2sp reads the reference every update so the plain run keeps it on the cpu
        ref_device = DEVICE if method["l2sp"] > 0 else "cpu"
        ref = {n:p.detach().to(ref_device,torch.bfloat16) for n,p in model.named_parameters()}
        setup["reference_seconds"] = time.perf_counter()-start
        setup["reference_bytes"] = sum(t.numel()*2 for t in ref.values())

    state = {"update":0,"history":[],"train_seconds":0.0,"peak_gb":0.0,"skipped":0,"export_seconds":0.0,"setup":setup,"displacement":{}}
    ckpt = run_dir/"latest.pt"
    if args.resume and ckpt.exists():
        state = load_checkpoint(ckpt,model,optimizer,scheduler,scaler)
        print(f"resumed {name} from update {state['update']}")
    if USE_AMP:
        torch.cuda.reset_peak_memory_stats()

    model.train()
    for update in range(state["update"],total):
        start = time.perf_counter()
        examples = [encode_example(tokenizer,r["query"],r["response"]) for r in rows[update*batch_size:(update+1)*batch_size]]
        total_tokens = sum(sum(l != -100 for l in e["labels"][1:]) for e in examples)
        loss_value = 0.0
        for i in range(0,batch_size,cfg["microbatch_size"]):
            batch = collate(examples[i:i+cfg["microbatch_size"]],tokenizer.pad_token_id)
            loss = micro_loss(model,batch,total_tokens)
            scaler.scale(loss).backward()
            loss_value += loss.item()
        scaler.unscale_(optimizer)
        record = {"update":update+1,"loss":loss_value,"lr":optimizer.param_groups[0]["lr"],"tokens":total_tokens}
        if method.get("l2sp",0) > 0:
            omega = 0.5*squared_distance(model,ref,method["l2sp"])
            record.update({"omega":omega,"penalty":method["l2sp"]*omega,"total_loss":loss_value+method["l2sp"]*omega})
        record["grad_norm"] = torch.nn.utils.clip_grad_norm_(params,cfg["max_grad_norm"]).item()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        record["skipped"] = scaler.get_scale() < scale_before
        state["skipped"] += record["skipped"]
        state["history"].append(record)
        state["update"] = update+1
        state["train_seconds"] += time.perf_counter()-start
        if USE_AMP:
            state["peak_gb"] = max(state["peak_gb"],torch.cuda.max_memory_allocated()/1e9)
        if (update+1)%25 == 0 or update == 0:
            print(f"update {update+1} loss {loss_value:.4f} lr {record['lr']:.2e} grad_norm {record['grad_norm']:.2f}")

        if update+1 == total//2 and args.method == "full":
            state["displacement"]["mid"] = squared_distance(model,ref)**0.5
            state["export_seconds"] += export(model,tokenizer,cfg,method,ref,run_dir,"mid")
        if update+1 in save_at:
            if update+1 == total and ref is not None:
                state["displacement"]["final"] = squared_distance(model,ref)**0.5
            state["checkpoint_seconds"] = save_checkpoint(ckpt,model,optimizer,scheduler,scaler,state)
        if args.stop_after and update+1 == args.stop_after:
            print(f"stopping after update {update+1}")
            return

    state["export_seconds"] += export(model,tokenizer,cfg,method,ref,run_dir,"final")
    final_path = run_dir/("final_model" if method["kind"] == "full" else "adapter.pt")
    summary = {
        "method":name,"updates":total,"examples_seen":total*batch_size,"learning_rate":method["lr"],
        "trainable_params":sum(p.numel() for p in params),"total_params":sum(p.numel() for p in model.parameters()),
        "train_seconds":state["train_seconds"],"peak_memory_gb":state["peak_gb"],"skipped_updates":state["skipped"],
        "export_seconds":state["export_seconds"],"final_checkpoint_bytes":folder_bytes(final_path),
        "resume_checkpoint_bytes":folder_bytes(ckpt),"displacement":state["displacement"],**state["setup"],
    }
    save_json(f"{results}/tables/{name}_run_summary.json",summary)
    save_json(f"{results}/tables/{name}_training_log.json",state["history"])
    print(f"finished {name} in {state['train_seconds']/3600:.2f} hours with {state['skipped']} skipped updates")


if __name__ == "__main__":
    main()
