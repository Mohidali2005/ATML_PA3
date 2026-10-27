import argparse
import json
import urllib.request
import zipfile
from collections import Counter
from functools import lru_cache

import numpy as np
from transformers import AutoTokenizer

from common.utils import load_yaml,repo_path,save_json,write_jsonl
from task2_vlm.metrics import normalize_answer

COCO_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
VQA_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
COCO_FILES = ["instances_train2014","instances_val2014","captions_train2014","captions_val2014"]
VQA_FILES = {
    "v2_Questions_Train_mscoco":"v2_OpenEnded_mscoco_train2014_questions.json",
    "v2_Questions_Val_mscoco":"v2_OpenEnded_mscoco_val2014_questions.json",
    "v2_Annotations_Train_mscoco":"v2_mscoco_train2014_annotations.json",
    "v2_Annotations_Val_mscoco":"v2_mscoco_val2014_annotations.json",
    "v2_Complementary_Pairs_Train_mscoco":"v2_mscoco_train2014_complementary_pairs.json",
    "v2_Complementary_Pairs_Val_mscoco":"v2_mscoco_val2014_complementary_pairs.json",
}
IRREGULAR = {"person":"people","bus":"buses","knife":"knives","sheep":"sheep","mouse":"mice","skis":"skis","scissors":"scissors","broccoli":"broccoli"}

norm = lru_cache(maxsize=None)(normalize_answer)


def image_key(split,image_id):
    """Name of an image in the feature store"""
    return f"{split}_{image_id}"


def plural(name):
    """Plural form of a category name"""
    if name in IRREGULAR:
        return IRREGULAR[name]
    return name+"es" if name.endswith(("ch","sh","ss","x")) else name+"s"


def download_raw(raw_dir):
    """Fetch and unpack the annotation files that are not on disk yet"""
    raw_dir.mkdir(parents=True,exist_ok=True)
    if not all((raw_dir/"annotations"/f"{f}.json").exists() for f in COCO_FILES):
        urllib.request.urlretrieve(COCO_ZIP,raw_dir/"coco.zip")
        with zipfile.ZipFile(raw_dir/"coco.zip") as z:
            z.extractall(raw_dir,[f"annotations/{f}.json" for f in COCO_FILES])
    for name,member in VQA_FILES.items():
        if not (raw_dir/member).exists():
            urllib.request.urlretrieve(f"{VQA_URL}{name}.zip",raw_dir/f"{name}.zip")
            with zipfile.ZipFile(raw_dir/f"{name}.zip") as z:
                z.extractall(raw_dir)


def read_captions(path):
    """Group the caption annotations by image"""
    by_image = {}
    for a in json.load(open(path))["annotations"]:
        by_image.setdefault(a["image_id"],[]).append((a["id"],a["caption"].strip()))
    return by_image


def pick_captions(rng,by_image,tokenizer,count,per_image,max_tokens,taken,split):
    """Choose images that have enough short captions and keep some captions of each"""
    ids = sorted(i for i in by_image if image_key(split,i) not in taken)
    rows = []
    for i in rng.permutation(ids):
        if len(rows) == count*per_image:
            break
        captions = by_image[int(i)]
        lengths = [len(x) for x in tokenizer([c for _,c in captions],add_special_tokens=False)["input_ids"]]
        valid = [c for c,n in zip(captions,lengths) if n <= max_tokens]
        if len(valid) >= per_image:
            for j in rng.choice(len(valid),per_image,replace=False):
                taken.add(image_key(split,int(i)))
                rows.append({"image":image_key(split,int(i)),"image_id":int(i),"caption_id":valid[j][0],"caption":valid[j][1]})
    return rows


def summarize_instances(path,image_ids):
    """Per image and category summary of the object annotations for the chosen images"""
    data = json.load(open(path))
    wanted = set(image_ids)
    sizes = {im["id"]:im["width"]*im["height"] for im in data["images"] if im["id"] in wanted}
    info = {i:{} for i in sorted(wanted)}
    for a in data["annotations"]:
        i = a["image_id"]
        if i in wanted:
            entry = info[i].setdefault(a["category_id"],{"crowd":False,"areas":[],"boxes":[]})
            if a["iscrowd"]:
                entry["crowd"] = True
            else:
                entry["areas"].append(a["area"]/sizes[i])
                entry["boxes"].append([round(v,1) for v in a["bbox"]])
    return info,{c["id"]:c["name"] for c in data["categories"]}


def grounding_candidates(info,names,min_area,count_range):
    """Images that make a clear presence positive or negative and a clear count for each category"""
    present = {c:[] for c in names}
    absent = {c:[] for c in names}
    counts = {c:{} for c in names}
    for image,cats in info.items():
        for c in names:
            e = cats.get(c)
            if e is None:
                absent[c].append(image)
            elif not e["crowd"]:
                if max(e["areas"]) >= min_area:
                    present[c].append(image)
                if min(e["areas"]) >= min_area and count_range[0] <= len(e["areas"]) <= count_range[1]:
                    counts[c].setdefault(len(e["areas"]),[]).append(image)
    return present,absent,counts


def article(name):
    """The indefinite article for a category name"""
    return "an" if name[0] in "aeiou" else "a"


def free(images,split,taken):
    """Images that no suite has used yet"""
    if taken is None:
        return images
    return [i for i in images if image_key(split,i) not in taken]


def grounding_item(split,info,image,cat,answer):
    """One image side of a grounding pair"""
    boxes = info[image].get(cat,{}).get("boxes",[])
    return {"image":image_key(split,image),"image_id":image,"answer":answer,"boxes":boxes}


def presence_pairs(rng,prefix,count,split,info,names,present,absent,taken):
    """Draw yes and no pairs with the category chosen uniformly"""
    cats = [c for c in names if present[c] and absent[c]]
    pairs = []
    while len(pairs) < count:
        c = cats[rng.integers(len(cats))]
        pos = free(present[c],split,taken)
        neg = free(absent[c],split,taken)
        if not pos or not neg:
            continue
        pos,neg = pos[rng.integers(len(pos))],neg[rng.integers(len(neg))]
        items = [grounding_item(split,info,pos,c,"yes"),grounding_item(split,info,neg,c,"no")]
        if taken is not None:
            taken.update(it["image"] for it in items)
        pairs.append({"pair_id":f"{prefix}_presence_{len(pairs)}","kind":"presence","category":names[c],
                      "question":f"Does this image contain {article(names[c])} {names[c]}?","items":[items[i] for i in rng.permutation(2)]})
    return pairs


def counting_pairs(rng,prefix,count,split,info,names,counts,taken):
    """Draw pairs of images with two different counts with the category chosen uniformly"""
    cats = [c for c in names if len(counts[c]) >= 2]
    pairs = []
    while len(pairs) < count:
        c = cats[rng.integers(len(cats))]
        avail = {n:free(v,split,taken) for n,v in counts[c].items()}
        avail = {n:v for n,v in avail.items() if v}
        if len(avail) < 2:
            continue
        a,b = rng.choice(sorted(avail),2,replace=False)
        items = [grounding_item(split,info,avail[int(n)][rng.integers(len(avail[int(n)]))],c,str(int(n))) for n in (a,b)]
        if taken is not None:
            taken.update(it["image"] for it in items)
        pairs.append({"pair_id":f"{prefix}_counting_{len(pairs)}","kind":"counting","category":names[c],
                      "question":f"How many {plural(names[c])} can you see?","items":[items[i] for i in rng.permutation(2)]})
    return pairs


def vqa_candidates(raw_dir,split,tokenizer,cfg):
    """Complementary question pairs that satisfy every selection rule"""
    d = cfg["data"]
    questions = {q["question_id"]:q for q in json.load(open(raw_dir/f"v2_OpenEnded_mscoco_{split}_questions.json"))["questions"]}
    annotations = {a["question_id"]:a for a in json.load(open(raw_dir/f"v2_mscoco_{split}_annotations.json"))["annotations"]}
    pairs = json.load(open(raw_dir/f"v2_mscoco_{split}_complementary_pairs.json"))
    dropped = Counter()
    kept = []
    for q1,q2 in pairs:
        a1,a2 = annotations[q1],annotations[q2]
        if questions[q1]["question"] != questions[q2]["question"]:
            dropped["question_text_differs"] += 1
        elif a1["answer_type"] != a2["answer_type"]:
            dropped["answer_type_differs"] += 1
        elif norm(a1["multiple_choice_answer"]) == norm(a2["multiple_choice_answer"]):
            dropped["same_answer"] += 1
        elif min(sum(norm(x["answer"]) == norm(a["multiple_choice_answer"]) for x in a["answers"]) for a in (a1,a2)) < d["min_agreement"]:
            dropped["low_agreement"] += 1
        else:
            kept.append((q1,q2))
    texts = sorted({questions[q1]["question"] for q1,_ in kept})
    answers = sorted({annotations[q]["multiple_choice_answer"] for pair in kept for q in pair})
    q_len = dict(zip(texts,[len(x) for x in tokenizer(texts,add_special_tokens=False)["input_ids"]]))
    a_len = dict(zip(answers,[len(x) for x in tokenizer(answers,add_special_tokens=False)["input_ids"]]))
    final = []
    for pair in kept:
        if q_len[questions[pair[0]]["question"]] > cfg["max_question_tokens"] or max(a_len[annotations[q]["multiple_choice_answer"]] for q in pair) > cfg["max_answer_tokens"]:
            dropped["too_long"] += 1
        else:
            final.append(pair)
    return final,questions,annotations,dict(dropped)


def vqa_pairs(rng,prefix,split,quota,candidates,questions,annotations,taken):
    """Draw pairs by answer type with every image used at most once in the suite"""
    used = set()
    pairs = []
    for t,count in quota.items():
        pool = [p for p in candidates if annotations[p[0]]["answer_type"] == t]
        got = 0
        for j in rng.permutation(len(pool)):
            if got == count:
                break
            q1,q2 = pool[int(j)]
            keys = [image_key(split,annotations[q]["image_id"]) for q in (q1,q2)]
            if any(k in used or (taken is not None and k in taken) for k in keys):
                continue
            used.update(keys)
            items = [{"image":image_key(split,annotations[q]["image_id"]),"image_id":annotations[q]["image_id"],"question_id":q,
                      "answer":annotations[q]["multiple_choice_answer"],"answers":[x["answer"] for x in annotations[q]["answers"]]} for q in (q1,q2)]
            pairs.append({"pair_id":f"{prefix}_vqa_{len(pairs)}","kind":"vqa","answer_type":t,"question":questions[q1]["question"],"items":items})
            got += 1
        assert got == count,f"only {got} {t} pairs available for {prefix}"
    if taken is not None:
        taken.update(used)
    return pairs


def pair_ids(pairs):
    """Compact identifiers of every selected pair"""
    return [{"pair_id":p["pair_id"],"question":p["question"],"images":[it["image_id"] for it in p["items"]],
             "question_ids":[it["question_id"] for it in p["items"] if "question_id" in it]} for p in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = cfg["data"]
    data_dir = repo_path(cfg["paths"]["data_dir"])
    ids_dir = cfg["paths"]["results_dir"]+"/ids"
    raw_dir = data_dir/"raw"
    rng = np.random.default_rng(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm"])
    download_raw(raw_dir)
    counts = {}

    train_captions = read_captions(raw_dir/"annotations/captions_train2014.json")
    caption_rows = pick_captions(rng,train_captions,tokenizer,d["train_images"],d["captions_per_image"],cfg["max_caption_tokens"],set(),"train2014")
    train_images = sorted({r["image_id"] for r in caption_rows})
    del train_captions

    info,names = summarize_instances(raw_dir/"annotations/instances_train2014.json",train_images)
    present,absent,count_images = grounding_candidates(info,names,d["min_area_fraction"],d["count_range"])
    pool = d["grounding_pool_pairs"]
    presence_pool = presence_pairs(rng,"train",pool,"train2014",info,names,present,absent,None)
    counting_pool = counting_pairs(rng,"train",pool,"train2014",info,names,count_images,None)

    candidates,questions,annotations,dropped = vqa_candidates(raw_dir,"train2014",tokenizer,cfg)
    counts["vqa_train_candidates"] = {"kept":len(candidates),"dropped":dropped}
    vqa_train = vqa_pairs(rng,"train","train2014",d["vqa_pairs"]["train"],candidates,questions,annotations,None)

    taken = set()
    val_captions = read_captions(raw_dir/"annotations/captions_val2014.json")
    val_info,val_names = summarize_instances(raw_dir/"annotations/instances_val2014.json",sorted(val_captions))
    present,absent,count_images = grounding_candidates(val_info,val_names,d["min_area_fraction"],d["count_range"])
    grounding = {}
    for suite in ("test","dev"):
        half = d["grounding_pairs"][suite]//2
        grounding[suite] = (presence_pairs(rng,suite,half,"val2014",val_info,val_names,present,absent,taken)
                            +counting_pairs(rng,suite,half,"val2014",val_info,val_names,count_images,taken))
    candidates,questions,annotations,dropped = vqa_candidates(raw_dir,"val2014",tokenizer,cfg)
    counts["vqa_val_candidates"] = {"kept":len(candidates),"dropped":dropped}
    vqa = {s:vqa_pairs(rng,s,"val2014",d["vqa_pairs"][s],candidates,questions,annotations,taken) for s in ("test","dev")}
    dev_captions = pick_captions(rng,val_captions,tokenizer,d["dev_caption_images"],1,cfg["max_caption_tokens"],taken,"val2014")

    suites = {"captions_train":caption_rows,"captions_dev":dev_captions,"presence_pool":presence_pool,"counting_pool":counting_pool,
              "vqa_train":vqa_train,"grounding_dev":grounding["dev"],"grounding_test":grounding["test"],"vqa_dev":vqa["dev"],"vqa_test":vqa["test"]}
    for name,rows in suites.items():
        write_jsonl(data_dir/f"{name}.jsonl",rows)
        save_json(f"{ids_dir}/{name}.json",[{"image_id":r["image_id"],"caption_id":r["caption_id"]} for r in rows] if name.startswith("captions") else pair_ids(rows))
        counts[name] = len(rows)

    held_out = [set(it["image"] for p in suites[s] for it in p["items"]) for s in ("grounding_dev","grounding_test","vqa_dev","vqa_test")]
    held_out.append({r["image"] for r in dev_captions})
    assert sum(len(s) for s in held_out) == len(set().union(*held_out)),"evaluation suites share an image"
    save_json(ids_dir+"/selection_counts.json",counts)
    print(json.dumps({k:v for k,v in counts.items() if isinstance(v,int)}))


if __name__ == "__main__":
    main()
