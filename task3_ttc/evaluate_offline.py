import argparse

import numpy as np
import pandas as pd

from common.utils import load_json,load_yaml,repo_path
from task3_ttc.selection import BASE_METHODS,VISION_METHODS,select_all,visconf_scores

COMPARISONS = [("SC","FirstAns"),("LogP","SC"),("SelfCert","SC"),("SelfCert+RWC","SC"),("Shortest","SC"),("Vision","SC"),("Vision+RWC","SC"),
               ("VisConf","SC"),("VisConf+RWC","SC"),("Tokenwise","SC"),("Tokenwise+RWC","SC"),("SelfCert+RWC","SelfCert"),("Vision+RWC","Vision"),
               ("VisConf+RWC","VisConf"),("Tokenwise+RWC","Tokenwise"),("SelfCert","LogP"),("SelfCert","Shortest"),("VisConf","Vision"),
               ("VisConf","SelfCert"),("VisConf+RWC","SelfCert+RWC"),("Tokenwise","VisConf"),("Tokenwise+RWC","VisConf+RWC")]


def load_arrays(cfg,name):
    """Processed arrays of one dataset and the saved rollout orders"""
    processed = repo_path(cfg["paths"]["processed_dir"])
    arrays = dict(np.load(processed/f"{name}_arrays.npz"))
    orders = np.load(repo_path(cfg["paths"]["results_dir"])/"ids"/f"orders_{name}.npy").astype(np.int64)
    tokens = dict(np.load(processed/"mathvista_tokens.npz")) if name == "mathvista" else None
    return arrays,orders,tokens


def run_dataset(cfg,name):
    """Choose a candidate for every method question order and budget"""
    arrays,orders,tokens = load_arrays(cfg,name)
    methods = BASE_METHODS+(VISION_METHODS if name == "mathvista" else [])
    budgets = cfg["budgets"]
    count = len(arrays["questions"])
    shape = (len(methods),count,len(budgets))
    accuracy,missed,cost = np.zeros(shape),np.zeros(shape),np.zeros(shape)
    oracle,groups = np.zeros(shape[1:]),np.zeros(shape[1:])
    picked = np.zeros(shape+(orders.shape[1],),dtype=np.int8)
    low,high,epsilon = (cfg["visconf"][k] for k in ("low_percentile","high_percentile","epsilon"))
    for q in range(count):
        for o in range(orders.shape[1]):
            for b,n in enumerate(budgets):
                idx = orders[q,o,:n]
                group,length,correct = arrays["group"][q,idx],arrays["tokens"][q,idx],arrays["correct"][q,idx]
                scores = {"L":arrays["L"][q,idx],"K":arrays["K"][q,idx]}
                if name == "mathvista":
                    confidence,tokenwise = visconf_scores(tokens["kl"],tokens["vision"],tokens["offsets"],arrays["rows"][q,idx],low,high,epsilon)
                    scores.update({"V":arrays["V"][q,idx],"Z":arrays["V"][q,idx]*confidence,"T":tokenwise})
                chosen = select_all(group,length,scores,methods)
                oracle[q,b] += correct.any()
                groups[q,b] += len(np.unique(group))
                for m,method in enumerate(methods):
                    position = chosen[method]
                    accuracy[m,q,b] += correct[position]
                    missed[m,q,b] += correct.any() and not correct[position]
                    cost[m,q,b] += length[0] if method == "FirstAns" else length.sum()
                    picked[m,q,b,o] = idx[position]
        if q%200 == 199:
            print(name,"evaluated",q+1,"questions")
    runs = orders.shape[1]
    return methods,{"accuracy":accuracy/runs,"missed":missed/runs,"cost":cost/runs,"oracle":oracle/runs,"groups":groups/runs,"picked":picked},arrays["questions"]


def long_table(name,methods,result,questions,splits,budgets):
    """One row per method question and budget"""
    split_of = {q:"development" for q in splits[name]["development"]}
    split_of.update({q:"evaluation" for q in splits[name]["evaluation"]})
    rows = []
    for m,method in enumerate(methods):
        for q,question in enumerate(questions):
            for b,n in enumerate(budgets):
                rows.append((name,question,split_of[question],method,n,result["accuracy"][m,q,b],result["missed"][m,q,b],result["cost"][m,q,b],result["oracle"][q,b],result["groups"][q,b]))
    return pd.DataFrame(rows,columns=["dataset","question","split","method","N","accuracy","missed","cost","oracle","groups"])


def summaries(table,samples,seed):
    """Method comparison failure decomposition and paired accuracy differences on the evaluation split"""
    evaluation = table[table.split == "evaluation"]
    grouped = evaluation.groupby(["dataset","method","N"],sort=False)
    comparison = grouped[["accuracy","oracle","groups","cost","missed"]].mean().reset_index()
    comparison["accuracy"] *= 100
    comparison["oracle"] *= 100
    comparison["selected_correct"] = comparison["accuracy"]
    comparison["available_but_missed"] = comparison["missed"]*100
    comparison["none_generated"] = 100-comparison["oracle"]
    rng = np.random.default_rng(seed)
    differences = []
    for name,sub in evaluation.groupby("dataset"):
        count = sub.question.nunique()
        index = rng.integers(0,count,size=(samples,count))
        for n,part in sub.groupby("N"):
            per_method = {m:g.sort_values("question").accuracy.to_numpy() for m,g in part.groupby("method")}
            for a,b in COMPARISONS:
                if a in per_method and b in per_method:
                    gap = (per_method[a]-per_method[b])*100
                    boot = gap[index].mean(axis=1)
                    differences.append((name,a,b,n,gap.mean(),*np.percentile(boot,[2.5,97.5])))
    return comparison,pd.DataFrame(differences,columns=["dataset","method","baseline","N","difference","low","high"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--datasets",nargs="*")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["paths"]["results_dir"])
    splits = load_json(results/"ids"/"splits.json")
    (results/"eval").mkdir(parents=True,exist_ok=True)
    (results/"tables").mkdir(parents=True,exist_ok=True)
    tables = []
    for name in args.datasets or cfg["datasets"]:
        methods,result,questions = run_dataset(cfg,name)
        np.savez_compressed(results/"eval"/f"{name}_selected.npz",methods=np.array(methods),budgets=np.array(cfg["budgets"]),picked=result["picked"])
        table = long_table(name,methods,result,questions,splits,cfg["budgets"])
        table.to_csv(results/"eval"/f"{name}_per_question.csv",index=False)
        tables.append(table)
    comparison,differences = summaries(pd.concat(tables),cfg["bootstrap_samples"],cfg["seed"])
    comparison.to_csv(results/"tables"/"method_comparison.csv",index=False)
    differences.to_csv(results/"tables"/"accuracy_differences.csv",index=False)
    print("wrote evaluation tables")


if __name__ == "__main__":
    main()
