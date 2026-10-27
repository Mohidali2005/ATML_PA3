import argparse
import time

import torch

from common.utils import load_yaml,repo_path,save_json,write_jsonl
from task2_vlm.evaluation import CONDITIONS,condition_entries,entry_generate,entry_nll,mismatch_mapping,pair_arrays,score_entries,summarize,swap_outcomes
from task2_vlm.metrics import bootstrap_interval
from task2_vlm.model import build_vlm,load_trainable
from task2_vlm.utils import DEVICE,FeatureStore,load_suites,load_tokenizer

SUITES = ["grounding_test","vqa_test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--work-dir")
    ap.add_argument("--features-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--tiny",action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_dir = repo_path(args.work_dir or cfg["paths"]["work_dir"])/"main"
    results = args.results_dir or cfg["paths"]["results_dir"]
    tokenizer = load_tokenizer(cfg)
    suites = load_suites(cfg,SUITES)
    store = FeatureStore(args.features_dir or cfg["paths"]["data_dir"]+"/features")
    vlm = build_vlm(cfg,tokenizer,args.tiny,lora=True)
    load_trainable(vlm,torch.load(run_dir/f"stage{len(cfg['stages'])}_weights.pt",map_location="cpu"))
    vlm.eval()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    samples,seed = cfg["eval"]["bootstrap_samples"],cfg["seed"]
    mappings = {s:mismatch_mapping(len(suites[s]),seed) for s in SUITES}
    save_json(f"{results}/ids/test_mismatch_mapping.json",mappings)
    scores,intervals,advantages,swaps = {},{},{},{}
    start = time.perf_counter()
    for suite in SUITES:
        pairs = suites[suite]
        arrays = {}
        scores[suite],intervals[suite] = {},{}
        for condition in CONDITIONS:
            entries = condition_entries(pairs,condition,mappings[suite])
            nlls = entry_nll(vlm,cfg,store,entries,cfg["eval"]["nll_batch_size"])
            predictions,seconds = entry_generate(vlm,cfg,store,entries)
            rows = score_entries(pairs,entries,predictions,nlls,seconds)
            write_jsonl(f"{results}/eval/{suite}_{condition}.jsonl",rows)
            scores[suite][condition] = summarize(rows)
            arrays[condition] = pair_arrays(rows)
            intervals[suite][condition] = {m:bootstrap_interval(v,samples,seed) for m,v in arrays[condition].items()}
            if condition == "swap":
                swaps[suite] = swap_outcomes(rows)
            print(f"{suite} {condition} accuracy {scores[suite][condition]['accuracy']:.3f} nll {scores[suite][condition]['nll']:.3f}")
        advantages[suite] = {c:{"accuracy":bootstrap_interval(arrays["correct"]["accuracy"]-arrays[c]["accuracy"],samples,seed),
                                "pair_accuracy":bootstrap_interval(arrays["correct"]["pair_accuracy"]-arrays[c]["pair_accuracy"],samples,seed),
                                "nll":bootstrap_interval(arrays[c]["nll"]-arrays["correct"]["nll"],samples,seed)} for c in CONDITIONS[1:]}
    out = {"scores":scores,"intervals":intervals,"correct_image_advantage":advantages,"swap_outcomes":swaps,
           "eval_seconds":time.perf_counter()-start,"peak_memory_gb":torch.cuda.max_memory_allocated()/1e9 if DEVICE == "cuda" else None}
    save_json(f"{results}/eval/final_scores.json",out)
    print(f"evaluated in {out['eval_seconds']:.0f} seconds")


if __name__ == "__main__":
    main()
