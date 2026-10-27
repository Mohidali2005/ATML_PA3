import time

import numpy as np
import torch

from task1_finetune.utils import collate
from task2_vlm.metrics import exact_match,normalize_answer,soft_accuracy
from task2_vlm.utils import DEVICE,amp,caption_examples,encode_example,encode_prompt

CONDITIONS = ["correct","swap","mismatch","none"]


def mismatch_mapping(num_pairs,seed):
    """Choose for every pair another pair whose images replace its own"""
    perm = np.random.default_rng(seed).permutation(num_pairs)
    for i in range(num_pairs):
        if perm[i] == i:
            j = (i+1)%num_pairs
            perm[i],perm[j] = perm[j],perm[i]
    assert all(perm[i] != i for i in range(num_pairs))
    return perm.tolist()


def condition_entries(pairs,condition,mapping):
    """One entry per question and image for an image condition"""
    entries = []
    for p,pair in enumerate(pairs):
        for i,item in enumerate(pair["items"]):
            source = {"correct":item,"swap":pair["items"][1-i],"mismatch":pairs[mapping[p]]["items"][i],"none":None}[condition]
            entries.append({"pair":p,"item":i,"kind":pair["kind"],"question":pair["question"],"answer":item["answer"],
                            "image":source["image"] if source else None})
    return entries


def entry_features(store,chunk):
    """Cached features of a batch or nothing when the batch has no image"""
    return store.get([e["image"] for e in chunk]) if chunk[0]["image"] is not None else None


@torch.no_grad()
def entry_nll(vlm,cfg,store,entries,batch_size):
    """Teacher forced answer loss of every entry with the end of turn token left out"""
    values = []
    for i in range(0,len(entries),batch_size):
        chunk = entries[i:i+batch_size]
        batch = collate([encode_example(vlm.tokenizer,cfg,e,e["image"] is not None) for e in chunk],vlm.tokenizer.pad_token_id)
        with amp():
            values += vlm.example_nll(batch,entry_features(store,chunk),include_end=False).tolist()
    return values


def entry_generate(vlm,cfg,store,entries):
    """Greedy answers with the time spent per answer"""
    predictions,seconds = [],[]
    for i in range(0,len(entries),cfg["eval"]["batch_size"]):
        chunk = entries[i:i+cfg["eval"]["batch_size"]]
        prompts = [encode_prompt(vlm.tokenizer,cfg,e,e["image"] is not None) for e in chunk]
        features = entry_features(store,chunk)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with amp():
            predictions += vlm.generate(prompts,features,cfg["eval"]["max_new_tokens"])
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        seconds += [(time.perf_counter()-start)/len(chunk)]*len(chunk)
    return predictions,seconds


def score_entries(pairs,entries,predictions,nlls,seconds=None):
    """Per answer scores against the original references"""
    rows = []
    for n,(e,pred) in enumerate(zip(entries,predictions)):
        pair = pairs[e["pair"]]
        item,partner = pair["items"][e["item"]],pair["items"][1-e["item"]]
        exact = exact_match(pred,item["answer"])
        rows.append({"pair_id":pair["pair_id"],"item":e["item"],"type":pair.get("answer_type",pair["kind"]),"question":pair["question"],
                     "image":e["image"],"reference":item["answer"],"partner_reference":partner["answer"],"prediction":pred,
                     "exact":exact,"accuracy":soft_accuracy(pred,item["answers"]) if "answers" in item else exact,
                     "partner_match":exact_match(pred,partner["answer"]),"empty":float(normalize_answer(pred) == ""),
                     "nll":nlls[n],"seconds":seconds[n] if seconds else None})
    return rows


def pair_arrays(rows):
    """Pair level accuracy and loss in the order of the pairs"""
    return {"accuracy":np.array([r["accuracy"] for r in rows]).reshape(-1,2).mean(axis=1),
            "nll":np.array([r["nll"] for r in rows]).reshape(-1,2).mean(axis=1),
            "pair_accuracy":np.array([r["exact"] for r in rows]).reshape(-1,2).all(axis=1).astype(float)}


def summarize(rows):
    """Metrics of one suite under one image condition"""
    pairs = np.array([r["exact"] for r in rows]).reshape(-1,2)
    words = [normalize_answer(r["prediction"]) for r in rows]
    same = np.array([a == b for a,b in zip(words[0::2],words[1::2])])
    types = sorted({r["type"] for r in rows})
    out = {"accuracy":float(np.mean([r["accuracy"] for r in rows])),
           "by_type":{t:float(np.mean([r["accuracy"] for r in rows if r["type"] == t])) for t in types},
           "pair_accuracy":float(pairs.all(axis=1).mean()),"identical_rate":float(same.mean()),
           "empty_rate":float(np.mean([r["empty"] for r in rows])),"nll":float(np.mean([r["nll"] for r in rows]))}
    if rows[0]["seconds"] is not None:
        out["seconds_per_answer"] = float(np.mean([r["seconds"] for r in rows]))
    return out


def swap_outcomes(rows):
    """How often an answer under a complementary swap keeps the original answer or follows the partner image"""
    n = len(rows)
    original = sum(r["exact"] for r in rows)
    partner = sum(r["partner_match"] for r in rows)
    return {"original_answer":original/n,"partner_answer":partner/n,"other":(n-original-partner)/n}


@torch.no_grad()
def caption_loss(vlm,cfg,store,rows,batch_size):
    """Mean reference caption loss over examples"""
    values = []
    examples = caption_examples(rows)
    for i in range(0,len(examples),batch_size):
        chunk = examples[i:i+batch_size]
        batch = collate([encode_example(vlm.tokenizer,cfg,e) for e in chunk],vlm.tokenizer.pad_token_id)
        with amp():
            values += vlm.example_nll(batch,entry_features(store,chunk)).tolist()
    return float(np.mean(values))


def dev_metrics(vlm,cfg,store,suites,mappings,generate):
    """Development caption loss and answer losses under correct swapped and missing images

    At stage endpoints it also generates answers under all four image conditions and a few captions
    """
    vlm.eval()
    out = {"caption_loss":caption_loss(vlm,cfg,store,suites["captions_dev"],cfg["eval"]["nll_batch_size"])}
    for suite in ("grounding_dev","vqa_dev"):
        pairs = suites[suite]
        nll = {}
        for c in ("correct","swap","none"):
            nll[c] = float(np.mean(entry_nll(vlm,cfg,store,condition_entries(pairs,c,mappings[suite]),cfg["eval"]["nll_batch_size"])))
        out[suite] = {"nll":nll,"delta_nll":nll["swap"]-nll["correct"]}
        if generate:
            out[suite]["accuracy"] = {}
            for c in CONDITIONS:
                entries = condition_entries(pairs,c,mappings[suite])
                preds,_ = entry_generate(vlm,cfg,store,entries)
                out[suite]["accuracy"][c] = summarize(score_entries(pairs,entries,preds,[0.0]*len(entries)))["accuracy"]
    if generate:
        rows = suites["captions_dev"][:cfg["eval"]["dev_generation_examples"]]
        batch = cfg["eval"]["batch_size"]
        texts = []
        for i in range(0,len(rows),batch):
            chunk = caption_examples(rows[i:i+batch])
            prompts = [encode_prompt(vlm.tokenizer,cfg,e) for e in chunk]
            with amp():
                texts += vlm.generate(prompts,entry_features(store,chunk),cfg["max_caption_tokens"])
        out["captions"] = [{"image":r["image"],"reference":r["caption"],"generated":t} for r,t in zip(rows,texts)]
    vlm.train()
    return out
