import argparse
import json
from importlib.metadata import version

import numpy as np
import pyarrow.parquet as pq

from common.utils import load_yaml,repo_path,save_json
from task3_ttc.answers import NO_ANSWER,VERIFIER,grade_math,grade_mathvista,group_math,group_mathvista


def flat_list(table,name):
    """Values of a list column as one flat array with the row offsets"""
    column = table[name].combine_chunks()
    return column.values.to_numpy(),column.offsets.to_numpy().astype(np.int64)


def row_means(values,offsets):
    """Mean of the values inside every row"""
    lengths = np.diff(offsets)
    return np.add.reduceat(values.astype(np.float64),offsets[:-1])/lengths


def canonical_rows(sample_ids,rollout_ids,rollouts):
    """Sorted question ids and the file row of every question and rollout"""
    questions = sorted(set(sample_ids))
    position = {q:i for i,q in enumerate(questions)}
    rows = np.full((len(questions),rollouts),-1,dtype=np.int64)
    for row,(q,r) in enumerate(zip(sample_ids,rollout_ids)):
        assert rows[position[q],r] == -1,"duplicate rollout"
        rows[position[q],r] = row
    assert (rows >= 0).all(),"missing rollout"
    return questions,rows


def split_questions(questions,seed,development):
    """Development and evaluation ids from one seeded permutation of the sorted ids"""
    permuted = [str(q) for q in np.random.default_rng(seed).permutation(questions)]
    return permuted[:development],permuted[development:]


def make_orders(count,orders,rollouts,seed):
    """Seeded permutations of the rollout ids for every question"""
    rng = np.random.default_rng(seed+1)
    base = np.tile(np.arange(rollouts),(count,orders,1))
    return rng.permuted(base,axis=2).astype(np.int8)


def check_math_source(cfg,questions,texts):
    """Number of questions whose text differs from the pinned source dataset row"""
    from datasets import load_dataset
    source = load_dataset(cfg["math_source"]["dataset"],revision=cfg["math_source"]["revision"],split="test")
    return sum(source[int(q.split(":")[1])]["problem"] != t for q,t in zip(questions,texts))


def build(cfg,name,meta):
    """Arrays of lengths scores answer groups and correctness for one dataset"""
    raw = repo_path(cfg["paths"]["raw_dir"])/cfg["datasets"][name]["folder"]
    rollouts = cfg["rollouts_per_question"]
    gens = pq.read_table(raw/"generations.parquet",columns=["sample_id","rollout_id","question_text","n_generated_text_tokens","hit_max_new_tokens","gold_answer","extracted_answer"])
    prob = pq.read_table(raw/"probability_metrics.parquet")
    sample_ids,rollout_ids = gens["sample_id"].to_pylist(),gens["rollout_id"].to_pylist()
    assert prob["sample_id"].to_pylist() == sample_ids and prob["rollout_id"].to_pylist() == rollout_ids,"probability rows are not aligned"
    questions,rows = canonical_rows(sample_ids,rollout_ids,rollouts)
    lengths = gens["n_generated_text_tokens"].to_numpy()
    checks = {"questions":len(questions),"rows":gens.num_rows,"retained_tokens":int(lengths.sum()),"zero_length_rollouts":int((lengths == 0).sum()),
              "capped_rollouts":int(gens["hit_max_new_tokens"].to_numpy().sum()),"null_extractions":gens["extracted_answer"].null_count}
    assert checks["zero_length_rollouts"] == 0
    logp,logp_off = flat_list(prob,"logp")
    kl,kl_off = flat_list(prob,"kl_u_p")
    assert (np.diff(logp_off) == lengths).all() and (np.diff(kl_off) == lengths).all(),"token lists do not match the lengths"
    scores = {"L":row_means(logp,logp_off),"K":row_means(kl,kl_off),"V":np.full(gens.num_rows,np.nan)}
    if name == "mathvista":
        attn = pq.read_table(raw/"attention_metrics.parquet")
        assert attn["sample_id"].to_pylist() == sample_ids and attn["rollout_id"].to_pylist() == rollout_ids
        image,image_off = flat_list(attn,"img_attn_total")
        prompt,_ = flat_list(attn,"prompt_text_attn_total")
        generated,_ = flat_list(attn,"generated_text_attn_total")
        count,count_off = flat_list(attn,"n_generated_text_tokens")
        assert (np.diff(image_off) == lengths).all() and (np.diff(count_off) == lengths).all()
        assert (count == np.arange(len(count))-np.repeat(count_off[:-1],lengths)).all(),"attention counts are not zero to n minus one"
        total = image.astype(np.float64)+prompt+generated
        checks["zero_attention_denominators"] = int((total <= 0).sum())
        vision = np.divide(image,total,out=np.zeros(len(total)),where=total > 0)
        scores["V"] = row_means(vision,image_off)
        np.savez(repo_path(cfg["paths"]["processed_dir"])/"mathvista_tokens.npz",kl=kl,vision=vision.astype(np.float32),offsets=kl_off)
    golds,extracted = gens["gold_answer"].to_pylist(),gens["extracted_answer"].to_pylist()
    count = len(questions)
    group,correct = np.full((count,rollouts),-1,dtype=np.int16),np.zeros((count,rollouts),dtype=bool)
    for i,q in enumerate(questions):
        answers = [extracted[r] for r in rows[i]]
        gold = golds[rows[i,0]]
        if name == "math":
            group[i] = group_math(answers)
            correct[i] = [grade_math(gold,a) for a in answers]
        else:
            group[i] = group_mathvista(answers,meta[q])
            correct[i] = [grade_mathvista(gold,a,meta[q]) for a in answers]
        if i%200 == 199:
            print(name,"graded",i+1,"questions")
    if name == "math":
        texts = gens["question_text"].to_pylist()
        checks["question_text_differences"] = int(check_math_source(cfg,questions,[texts[rows[i,0]] for i in range(count)]))
    if name == "math":
        checks["math_verify_timeouts"] = VERIFIER.timeouts
    checks["no_answer_rollouts"] = int((group == -1).sum())
    checks["mean_distinct_groups"] = float(np.mean([len(set(g[g >= 0])) for g in group]))
    checks["mean_accuracy"] = float(correct.mean())
    checks["oracle_coverage"] = float(correct.any(axis=1).mean())
    arrays = {"questions":np.array(questions),"rows":rows,"tokens":lengths[rows].astype(np.int32),"capped":gens["hit_max_new_tokens"].to_numpy()[rows],
              "group":group,"correct":correct,"L":scores["L"][rows].astype(np.float32),"K":scores["K"][rows].astype(np.float32),"V":scores["V"][rows].astype(np.float32)}
    return arrays,checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    processed = repo_path(cfg["paths"]["processed_dir"])
    processed.mkdir(parents=True,exist_ok=True)
    results = cfg["paths"]["results_dir"]
    ids_dir = repo_path(results)/"ids"
    ids_dir.mkdir(parents=True,exist_ok=True)
    meta = json.load(open(repo_path(cfg["paths"]["raw_dir"])/"testmini.json",encoding="utf-8"))
    splits,all_checks = {},{"math_verify_version":version("math-verify"),"no_answer_label":NO_ANSWER}
    for name in cfg["datasets"]:
        arrays,checks = build(cfg,name,meta)
        np.savez(processed/f"{name}_arrays.npz",**arrays)
        development,evaluation = split_questions(list(arrays["questions"]),cfg["seed"],cfg["dev_questions"])
        splits[name] = {"development":development,"evaluation":evaluation}
        orders = make_orders(len(arrays["questions"]),cfg["num_orders"],cfg["rollouts_per_question"],cfg["seed"])
        np.save(ids_dir/f"orders_{name}.npy",orders)
        all_checks[name] = checks
        print(name,checks)
    splits["live"] = splits["math"]["evaluation"][:cfg["live_questions"]]
    save_json(f"{results}/ids/splits.json",splits)
    save_json(f"{results}/tables/data_checks.json",all_checks)


if __name__ == "__main__":
    main()
