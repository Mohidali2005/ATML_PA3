import argparse

import pandas as pd
import torch

from common.utils import load_json,load_yaml,repo_path
from task1_finetune.utils import load_model,load_tokenizer

# each entry is the condition name and the file that holds its down_proj updates
SOURCES = {
    "full_mid":"full/diag_mid.pt","full":"full/diag_final.pt","l2sp":"l2sp/diag_final.pt",
    "lora4":"lora4/diag_final.pt","lora16":"lora16/diag_final.pt","oplora":"oplora/diag_final.pt",
}


def cache_inputs(cfg,model,tokenizer,probes):
    """Run every probe prompt through the starting model and keep the down_proj input at the last token"""
    store = {layer:[] for layer in cfg["diag_layers"]}
    hooks = []
    for layer in cfg["diag_layers"]:
        def hook(module,inputs,layer=layer):
            store[layer].append(inputs[0][0,-1].detach().float())
        hooks.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
    for probe in probes:
        ids = tokenizer(probe["text"],return_tensors="pt",add_special_tokens=False)
        with torch.no_grad():
            model(**ids)
    for h in hooks:
        h.remove()
    return {layer:torch.stack(rows) for layer,rows in store.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--tiny",action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    work_dir = repo_path(args.work_dir or cfg["paths"]["work_dir"])
    probes = load_json(cfg["paths"]["data_dir"]+"/probe_prompts.json")
    is_math = torch.tensor([p["domain"] == "math" for p in probes])
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg,torch.float32,args.tiny).eval()
    inputs = cache_inputs(cfg,model,tokenizer,probes)

    rows = []
    for layer in cfg["diag_layers"]:
        w0 = model.model.layers[layer].mlp.down_proj.weight.detach().float()
        U,S,Vh = torch.linalg.svd(w0,full_matrices=False)
        Uk = U[:,:min(cfg["protect_rank"],len(S))]
        z = inputs[layer]
        base_out = (z@w0.T).norm(dim=1)
        for condition,file in SOURCES.items():
            if not (work_dir/file).exists():
                continue
            delta = torch.load(work_dir/file,weights_only=False)[layer].float()
            ratio = (z@delta.T).norm(dim=1)/(base_out+1e-8)
            rows.append({"condition":condition,"layer":layer,
                         "rho":((Uk.T@delta).pow(2).sum()/delta.pow(2).sum()).item(),
                         "eta":(delta.norm()/w0.norm()).item(),
                         "R_math":ratio[is_math].mean().item(),"R_general":ratio[~is_math].mean().item()})
    table = pd.DataFrame(rows)
    out = repo_path(cfg["paths"]["results_dir"]+"/tables/diagnostics.csv")
    out.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(out,index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
