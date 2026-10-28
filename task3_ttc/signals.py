import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common.utils import load_json,load_yaml,repo_path,save_json
from task3_ttc.evaluate_offline import load_arrays
from task3_ttc.selection import token_confidence


def rank_correlation(a,b):
    """Rank correlation of two arrays, not a number when either one is constant"""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])


def quartile_gap(score,correct):
    """Correctness of the top quartile of a score minus that of the bottom quartile"""
    k = len(score)//4
    order = np.argsort(-score,kind="stable")
    top,bottom = correct[order[:k]].mean(),correct[order[-k:]].mean()
    return top,bottom


def question_signals(arrays,tokens,q,cfg):
    """Per rollout signals of one question with all candidates revealed"""
    signals = {"Length":arrays["tokens"][q].astype(float),"LogP":arrays["L"][q],"SelfCert":arrays["K"][q]}
    stats = None
    if tokens is not None:
        low,high,epsilon = (cfg["visconf"][k] for k in ("low_percentile","high_percentile","epsilon"))
        rows = arrays["rows"][q]
        confidence = token_confidence(tokens["kl"],tokens["offsets"],rows,low,high,epsilon)
        vision = [tokens["vision"][tokens["offsets"][r]:tokens["offsets"][r+1]].astype(np.float64) for r in rows]
        mean_c = np.array([c.mean() for c in confidence])
        mean_v = arrays["V"][q].astype(np.float64)
        token_z = np.array([(c*v).mean() for c,v in zip(confidence,vision)])
        signals.update({"Vision":mean_v,"Confidence":mean_c,"VisConf":mean_v*mean_c,"Tokenwise":token_z})
        correlation = np.array([np.corrcoef(c,v)[0,1] if np.ptp(c) > 0 and np.ptp(v) > 0 else np.nan for c,v in zip(confidence,vision)])
        covariance = np.array([(c*v).mean()-c.mean()*v.mean() for c,v in zip(confidence,vision)])
        stats = {"correlation":correlation,"identity_error":np.abs(token_z-mean_v*mean_c-covariance).max()}
    return signals,stats


def bootstrap_mean(values,samples,seed):
    """Mean and 95 percent interval over questions"""
    values = np.asarray(values,dtype=float)
    index = np.random.default_rng(seed).integers(0,len(values),size=(samples,len(values)))
    return float(values.mean()),*np.percentile(values[index].mean(axis=1),[2.5,97.5])


def analyze_dataset(cfg,name,evaluation):
    """Quartile gaps rank correlations and the confidence vision relationship at the full candidate pool"""
    arrays,_,tokens = load_arrays(cfg,name)
    position = {q:i for i,q in enumerate(arrays["questions"])}
    gaps,correlations,vision_stats,agreement = {},{},{"correlation":[],"identity_error":0.0},{"rho":[],"same_top":[]}
    mixed = []
    for question in evaluation:
        q = position[question]
        correct = arrays["correct"][q].astype(float)
        signals,stats = question_signals(arrays,tokens,q,cfg)
        mixed.append(0 < correct.sum() < len(correct))
        for signal,score in signals.items():
            top,bottom = quartile_gap(score,correct)
            gaps.setdefault(signal,[]).append((top,bottom))
            if signal != "Length":
                correlations.setdefault(("Length",signal),[]).append(rank_correlation(signals["Length"],score))
            correlations.setdefault((signal,"Correct"),[]).append(rank_correlation(score,correct))
        if stats is not None:
            vision_stats["correlation"].append(np.nanmean(stats["correlation"]))
            vision_stats["identity_error"] = max(vision_stats["identity_error"],stats["identity_error"])
            agreement["rho"].append(rank_correlation(signals["VisConf"],signals["Tokenwise"]))
            agreement["same_top"].append(float(np.argmax(signals["VisConf"]) == np.argmax(signals["Tokenwise"])))
    mixed = np.array(mixed)
    quartiles = []
    for signal,pairs in gaps.items():
        pairs = np.array(pairs)
        pairs = 100*pairs
        difference = pairs[:,0]-pairs[:,1]
        mean_all,low,high = bootstrap_mean(difference,cfg["bootstrap_samples"],cfg["seed"])
        quartiles.append((name,signal,mean_all,low,high,difference[mixed].mean(),int(mixed.sum()),pairs[:,0].mean(),pairs[:,1].mean()))
    table = pd.DataFrame(quartiles,columns=["dataset","signal","gap_all","low","high","gap_mixed","mixed_questions","top_accuracy","bottom_accuracy"])
    rho = pd.DataFrame([(name,a,b,np.nanmean(v),int(np.sum(~np.isnan(v)))) for (a,b),v in correlations.items()],columns=["dataset","signal","versus","mean_rho","questions"])
    extra = {}
    if tokens is not None:
        values = np.array(vision_stats["correlation"])
        extra = {"confidence_vision_correlation":{"mean":float(np.nanmean(values)),"median":float(np.nanmedian(values)),"negative_share":float(np.nanmean(values < 0)),
                 "quantiles":[float(x) for x in np.nanpercentile(values,[5,25,75,95])],"questions":int(np.sum(~np.isnan(values)))},
                 "covariance_identity_max_error":vision_stats["identity_error"],
                 "visconf_versus_tokenwise":{"mean_rank_correlation":float(np.nanmean(agreement["rho"])),"same_top_rollout_share":float(np.mean(agreement["same_top"]))}}
        np.save(repo_path(cfg["paths"]["results_dir"])/"eval"/"mathvista_confidence_vision_correlation.npy",values)
    return table,rho,extra


def pick_trajectories(arrays,tokens,evaluation,cfg):
    """Illustrative questions by a fixed rule

    agree is the first evaluation question with mixed correctness where the highest self certainty rollout and the highest VisConf rollout are the same
    disagree is the first one where they differ and exactly one of the two is correct
    """
    position = {q:i for i,q in enumerate(arrays["questions"])}
    chosen = {}
    for question in evaluation:
        q = position[question]
        correct = arrays["correct"][q]
        if not 0 < correct.sum() < len(correct):
            continue
        signals,_ = question_signals(arrays,tokens,q,cfg)
        by_k,by_z = int(np.argmax(signals["SelfCert"])),int(np.argmax(signals["VisConf"]))
        if "agree" not in chosen and by_k == by_z:
            chosen["agree"] = {"question":question,"selfcert_rollout":by_k,"visconf_rollout":by_z,"selfcert_correct":bool(correct[by_k]),"visconf_correct":bool(correct[by_z])}
        if "disagree" not in chosen and by_k != by_z and correct[by_k] != correct[by_z]:
            chosen["disagree"] = {"question":question,"selfcert_rollout":by_k,"visconf_rollout":by_z,"selfcert_correct":bool(correct[by_k]),"visconf_correct":bool(correct[by_z])}
        if len(chosen) == 2:
            break
    return chosen


def fusion_table(results):
    """Highest score and rank weighted consensus under summary level and tokenwise fusion"""
    comparison = pd.read_csv(results/"tables"/"method_comparison.csv")
    differences = pd.read_csv(results/"tables"/"accuracy_differences.csv")
    rows = []
    for n in (4,8,16,32):
        for label,summary,token in (("highest score","VisConf","Tokenwise"),("RWC","VisConf+RWC","Tokenwise+RWC")):
            a = comparison[(comparison.dataset == "mathvista") & (comparison.method == summary) & (comparison.N == n)].accuracy.iloc[0]
            b = comparison[(comparison.dataset == "mathvista") & (comparison.method == token) & (comparison.N == n)].accuracy.iloc[0]
            d = differences[(differences.dataset == "mathvista") & (differences.method == token) & (differences.baseline == summary) & (differences.N == n)].iloc[0]
            rows.append((n,label,a,b,d["difference"],d["low"],d["high"]))
    return pd.DataFrame(rows,columns=["N","aggregation","visconf","tokenwise","difference","low","high"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["paths"]["results_dir"])
    splits = load_json(results/"ids"/"splits.json")
    quartiles,correlations,extra = [],[],{}
    for name in cfg["datasets"]:
        table,rho,more = analyze_dataset(cfg,name,splits[name]["evaluation"])
        quartiles.append(table)
        correlations.append(rho)
        extra[name] = more
    pd.concat(quartiles).to_csv(results/"tables"/"signal_quartiles.csv",index=False)
    pd.concat(correlations).to_csv(results/"tables"/"signal_correlations.csv",index=False)
    arrays,_,tokens = load_arrays(cfg,"mathvista")
    extra["trajectories"] = pick_trajectories(arrays,tokens,splits["mathvista"]["evaluation"],cfg)
    save_json(results/"tables"/"signal_statistics.json",extra)
    fusion_table(results).to_csv(results/"tables"/"fusion_comparison.csv",index=False)
    print("wrote signal tables")


if __name__ == "__main__":
    main()
