import argparse

import numpy as np
import pandas as pd

from common.utils import load_json,load_yaml,repo_path,save_json
from task3_ttc.evaluate_offline import load_arrays
from task3_ttc.selection import rwc_vote,visconf_scores,vote_winner


def checkpoint_features(cfg,name):
    """Votes consensus margin and cost at every checkpoint for every question and order, from revealed candidates only"""
    arrays,orders,tokens = load_arrays(cfg,name)
    checks = cfg["adaptive"]["check_points"]
    count,runs = len(arrays["questions"]),orders.shape[1]
    shape = (count,runs,len(checks)+1)
    feature = {k:np.zeros(shape) for k in ("correct","leader_votes","clear_leader","same_winner","margin","cost","rollouts")}
    low,high,epsilon = (cfg["visconf"][k] for k in ("low_percentile","high_percentile","epsilon"))
    for q in range(count):
        for o in range(runs):
            for c,n in enumerate(checks+[orders.shape[2]]):
                idx = orders[q,o,:n]
                group,length = arrays["group"][q,idx],arrays["tokens"][q,idx]
                winner = vote_winner(group)
                labels,votes = np.unique(group,return_counts=True)
                ranked = np.sort(votes)[::-1]
                feature["correct"][q,o,c] = arrays["correct"][q,idx[winner]]
                feature["leader_votes"][q,o,c] = ranked[0]
                feature["clear_leader"][q,o,c] = (len(ranked) == 1 or ranked[0] > ranked[1]) and group[winner] != -1
                feature["cost"][q,o,c],feature["rollouts"][q,o,c] = length.sum(),n
                if c < len(checks):
                    if name == "math":
                        scores = arrays["K"][q,idx]
                    else:
                        confidence,_ = visconf_scores(tokens["kl"],tokens["vision"],tokens["offsets"],arrays["rows"][q,idx],low,high,epsilon)
                        scores = arrays["V"][q,idx]*confidence
                    position,margin = rwc_vote(scores,group)
                    feature["same_winner"][q,o,c] = group[position] == group[winner]
                    feature["margin"][q,o,c] = margin
        if q%200 == 199:
            print(name,"revealed",q+1,"questions")
    return arrays,orders,feature


def run_policy(feature,rows,tau,gamma,checks):
    """Accuracy cost and stopping point of one controller on the chosen questions

    tau none means the policy never stops early and gamma none leaves out the quality condition
    """
    f = {k:v[rows] for k,v in feature.items()}
    final = len(checks)
    stop = np.full(f["correct"].shape[:2],final)
    if tau is not None:
        for c in reversed(range(len(checks))):
            ok = f["clear_leader"][:,:,c] > 0
            ok &= f["leader_votes"][:,:,c] >= tau*checks[c]
            if gamma is not None:
                ok &= (f["same_winner"][:,:,c] > 0) & (f["margin"][:,:,c] >= gamma)
            stop = np.where(ok,c,stop)
    pick = lambda key:np.take_along_axis(f[key],stop[:,:,None],axis=2)[:,:,0]
    return {"correct":pick("correct"),"cost":pick("cost"),"rollouts":pick("rollouts"),"stop":stop}


def summarize(result,checks):
    """Mean accuracy cost rollouts and the share of stops at every checkpoint"""
    out = {"accuracy":100*result["correct"].mean(),"cost":result["cost"].mean(),"rollouts":result["rollouts"].mean()}
    for c,n in enumerate(checks+[32]):
        out[f"stop_{n}"] = float((result["stop"] == c).mean())
    return out


def setting_grid(cfg):
    """Every controller setting with its family name"""
    grid = [("Always 32",None,None)]
    grid += [("Agreement",tau,None) for tau in cfg["adaptive"]["tau"]]
    grid += [("Quality aware",tau,gamma) for tau in cfg["adaptive"]["tau"] for gamma in cfg["adaptive"]["gamma"]]
    return grid


def select_settings(development,tolerance):
    """Cheapest setting per family whose accuracy stays within the tolerance of fixed 32, ties to higher accuracy then smaller tau and gamma"""
    reference = development[development.family == "Always 32"].accuracy.iloc[0]
    chosen = {}
    for family in ("Agreement","Quality aware"):
        eligible = development[(development.family.isin([family,"Always 32"])) & (development.accuracy >= reference-100*tolerance-1e-9)].copy()
        eligible["cost_key"] = eligible.cost.round(6)
        eligible["tau_key"] = eligible.tau.fillna(np.inf)
        eligible["gamma_key"] = eligible.gamma.fillna(-1.0)
        best = eligible.sort_values(["cost_key","accuracy","tau_key","gamma_key"],ascending=[True,False,True,True]).iloc[0]
        chosen[family] = {"tau":None if np.isnan(best.tau) else float(best.tau),"gamma":None if np.isnan(best.gamma) else float(best.gamma),
                          "development_accuracy":float(best.accuracy),"development_cost":float(best.cost),"fixed_32_accuracy":float(reference)}
    return chosen


def question_rows(questions,ids):
    """Array positions of the given question ids"""
    position = {q:i for i,q in enumerate(questions)}
    return np.array([position[q] for q in ids])


def bootstrap_gap(a,b,samples,seed):
    """Paired question level difference of two per question arrays with a 95 percent interval"""
    gap = a-b
    index = np.random.default_rng(seed).integers(0,len(gap),size=(samples,len(gap)))
    return float(gap.mean()),*np.percentile(gap[index].mean(axis=1),[2.5,97.5])


def analyze_dataset(cfg,name,splits,results):
    """Development grid selection frozen settings and held out results of the adaptive controllers"""
    checks = cfg["adaptive"]["check_points"]
    arrays,orders,feature = checkpoint_features(cfg,name)
    questions = arrays["questions"]
    dev_rows,eval_rows = (question_rows(questions,splits[name][k]) for k in ("development","evaluation"))
    grid = setting_grid(cfg)
    development = pd.DataFrame([{"family":f,"tau":t,"gamma":g,**summarize(run_policy(feature,dev_rows,t,g,checks),checks)} for f,t,g in grid])
    chosen = select_settings(development,cfg["adaptive"]["tolerance"])
    save_json(results/"tables"/f"adaptive_selected_{name}.json",chosen)
    development.assign(dataset=name).to_csv(results/"tables"/f"adaptive_development_grid_{name}.csv",index=False)
    rows,per_question = [],{}
    fixed = run_policy(feature,eval_rows,None,None,checks)
    per_question["Fixed 32"] = fixed["correct"].mean(axis=1)
    for c,n in enumerate(checks+[32]):
        correct = feature["correct"][eval_rows][:,:,c]
        cost = feature["cost"][eval_rows][:,:,c]
        rows.append({"policy":f"Fixed {n}","tau":None,"gamma":None,"accuracy":100*correct.mean(),"cost":cost.mean(),"rollouts":float(n)})
        per_question[f"Fixed {n}"] = correct.mean(axis=1)
    early = {}
    for family,setting in chosen.items():
        result = run_policy(feature,eval_rows,setting["tau"],setting["gamma"] if family == "Quality aware" else None,checks)
        rows.append({"policy":family,"tau":setting["tau"],"gamma":setting["gamma"],**summarize(result,checks)})
        per_question[family] = result["correct"].mean(axis=1)
        early[family] = result
    table = pd.DataFrame(rows).assign(dataset=name)
    gaps = []
    for policy in table.policy:
        mean,low,high = bootstrap_gap(per_question[policy],per_question["Fixed 32"],cfg["bootstrap_samples"],cfg["seed"])
        gaps.append((100*mean,100*low,100*high))
    table[["difference_vs_fixed_32","low","high"]] = gaps
    save_json(results/"tables"/f"adaptive_premature_{name}.json",premature_cases(early,feature,eval_rows,questions,orders,checks))
    return table,early,feature,eval_rows,arrays


def premature_cases(early,feature,eval_rows,questions,orders,checks):
    """First early stop of each controller in evaluation order that returned a wrong answer where fixed 32 would have been correct"""
    fixed = feature["correct"][eval_rows][:,:,-1]
    cases = {}
    for family,result in early.items():
        wrong = (result["stop"] < len(checks)) & (result["correct"] == 0) & (fixed == 1)
        hits = np.argwhere(wrong)
        if len(hits):
            i,o = hits[0]
            n = checks[result["stop"][i,o]]
            cases[family] = {"question":str(questions[eval_rows[i]]),"order":int(o),"stop_at":int(n),"candidates":[int(x) for x in orders[eval_rows[i],o,:n]]}
    return cases


def early_stop_analysis(name,early,feature,eval_rows):
    """Incorrect early stops and the link between stopping and how many candidates in the full pool are correct"""
    summary = []
    final = feature["correct"].shape[2]-1
    for family,result in early.items():
        stopped = result["stop"] < final
        wrong = stopped & (result["correct"] == 0)
        recovered = wrong & (feature["correct"][eval_rows][:,:,-1] == 1)
        summary.append((name,family,float(stopped.mean()),float(wrong.mean()),float(recovered.mean()),float(wrong.sum()),float(recovered.sum())))
    return pd.DataFrame(summary,columns=["dataset","policy","early_stop_share","wrong_early_stop_share","wrong_but_fixed_32_correct_share","wrong_early_stops","recovered_by_fixed_32"])


def retrospective(cfg,name,early,arrays_correct,eval_rows):
    """Early stop rate and accuracy by the share of correct rollouts in the full pool of each question"""
    share = arrays_correct[eval_rows].mean(axis=1)
    edges = [-0.01,0.0,0.25,0.5,0.75,0.999,1.0]
    labels = ["none correct","up to 25%","25 to 50%","50 to 75%","over 75%","all correct"]
    bucket = pd.cut(share,edges,labels=labels)
    rows = []
    for family,result in early.items():
        stopped = (result["stop"] < len(cfg["adaptive"]["check_points"])).mean(axis=1)
        for label in labels:
            mask = np.asarray(bucket == label)
            if mask.any():
                rows.append((name,family,label,int(mask.sum()),float(stopped[mask].mean()),float(100*result["correct"][mask].mean()),float(result["rollouts"][mask].mean())))
    return pd.DataFrame(rows,columns=["dataset","policy","pool_correct_share","questions","early_stop_rate","accuracy","mean_rollouts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--datasets",nargs="*")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["paths"]["results_dir"])
    splits = load_json(results/"ids"/"splits.json")
    tables,stops,links = [],[],[]
    for name in args.datasets or cfg["datasets"]:
        table,early,feature,eval_rows,arrays = analyze_dataset(cfg,name,splits,results)
        tables.append(table)
        stops.append(early_stop_analysis(name,early,feature,eval_rows))
        links.append(retrospective(cfg,name,early,arrays["correct"],eval_rows))
    pd.concat(tables).to_csv(results/"tables"/"adaptive_results.csv",index=False)
    pd.concat(stops).to_csv(results/"tables"/"adaptive_early_stops.csv",index=False)
    pd.concat(links).to_csv(results/"tables"/"adaptive_retrospective.csv",index=False)
    print("wrote adaptive tables")


if __name__ == "__main__":
    main()
