import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import load_json,load_yaml,repo_path,save_json
from task3_ttc.evaluate_offline import load_arrays
from task3_ttc.selection import rwc_vote,token_confidence,visconf_scores,vote_winner

COLORS = {"FirstAns":"gray","SC":"black","LogP":"goldenrod","SelfCert":"seagreen","SelfCert+RWC":"darkgreen","Shortest":"sienna","Vision":"orchid",
          "Vision+RWC":"purple","VisConf":"steelblue","VisConf+RWC":"navy","Tokenwise":"darkorange","Tokenwise+RWC":"firebrick"}
NAMES = {"math":"MATH (Qwen2.5-3B)","mathvista":"MathVista (Qwen2.5-VL-3B)"}


def smooth(values,window):
    """Moving average over a window of tokens"""
    return np.convolve(values,np.ones(window)/window,mode="valid")


def sampling_curves(comparison,figures):
    """Accuracy of every method and oracle coverage against the candidate budget and against token cost"""
    fig,axes = plt.subplots(2,2,figsize=(13,9.5))
    for row,name in zip(axes,NAMES):
        data = comparison[comparison.dataset == name]
        for method,part in data.groupby("method",sort=False):
            for ax,x in zip(row,("N","cost")):
                ax.plot(part[x],part.accuracy,marker="o",markersize=4,linewidth=1.6,color=COLORS[method],label=method)
        oracle = data[data.method == "SC"]
        for ax,x in zip(row,("N","cost")):
            ax.plot(oracle[x],oracle.oracle,marker="s",markersize=4,linestyle="--",linewidth=1.8,color="dimgray",label="Oracle coverage")
            ax.set_xscale("log",base=2 if x == "N" else 10)
            ax.grid(alpha=0.25)
            ax.set_ylabel("Accuracy (%)",fontsize=12)
            ax.tick_params(labelsize=11)
        row[0].set_xticks([1,2,4,8,16,32])
        row[0].set_xticklabels([1,2,4,8,16,32])
        row[0].set_xlabel("Candidates N",fontsize=12)
        row[1].set_xlabel("Mean generated tokens per question",fontsize=12)
        row[0].set_title(f"{NAMES[name]}: accuracy against candidates",fontsize=12)
        row[1].set_title(f"{NAMES[name]}: accuracy against cost",fontsize=12)
    handles,labels = axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=5,fontsize=11,frameon=False)
    fig.tight_layout(rect=(0,0.08,1,1))
    fig.savefig(figures/"sampling_curves.png",dpi=150)
    plt.close(fig)


def failure_decomposition(comparison,figures):
    """Stacked shares of selected correct available but missed and none generated for self consistency plus distinct answer groups"""
    fig,axes = plt.subplots(1,3,figsize=(15,4.8))
    for ax,name in zip(axes[:2],NAMES):
        data = comparison[(comparison.dataset == name) & (comparison.method == "SC")]
        x = np.arange(len(data))
        ax.bar(x,data.selected_correct,color="seagreen",label="Selected correct")
        ax.bar(x,data.available_but_missed,bottom=data.selected_correct,color="darkorange",label="Correct available but missed")
        ax.bar(x,data.none_generated,bottom=data.selected_correct+data.available_but_missed,color="firebrick",label="No correct candidate")
        ax.set_xticks(x)
        ax.set_xticklabels(data.N)
        ax.set_xlabel("Candidates N",fontsize=12)
        ax.set_ylabel("Share of questions (%)",fontsize=12)
        ax.set_title(f"{NAMES[name]}, self consistency",fontsize=12)
        ax.tick_params(labelsize=11)
    axes[0].legend(loc="upper center",bbox_to_anchor=(0.5,-0.18),ncol=1,fontsize=10,frameon=False)
    for name,color in zip(NAMES,("steelblue","darkorange")):
        data = comparison[(comparison.dataset == name) & (comparison.method == "SC")]
        axes[2].plot(data.N,data.groups,marker="o",color=color,label=NAMES[name])
    axes[2].set_xscale("log",base=2)
    axes[2].set_xticks([1,2,4,8,16,32])
    axes[2].set_xticklabels([1,2,4,8,16,32])
    axes[2].set_xlabel("Candidates N",fontsize=12)
    axes[2].set_ylabel("Distinct answer groups",fontsize=12)
    axes[2].set_title("Distinct answers among candidates",fontsize=12)
    axes[2].legend(loc="upper left",fontsize=10,frameon=False)
    axes[2].grid(alpha=0.25)
    axes[2].tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(figures/"failure_decomposition.png",dpi=150)
    plt.close(fig)


def signal_figure(quartiles,results,figures):
    """Top minus bottom quartile correctness per signal and the confidence vision correlation histogram"""
    fig,axes = plt.subplots(1,3,figsize=(16,5),gridspec_kw={"width_ratios":[3,7,5]})
    for ax,name in zip(axes[:2],NAMES):
        data = quartiles[quartiles.dataset == name]
        x = np.arange(len(data))
        ax.bar(x,data.gap_all*100,color="steelblue",yerr=[(data.gap_all-data.low)*100,(data.high-data.gap_all)*100],capsize=3)
        ax.axhline(0,color="black",linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(data.signal,rotation=35,ha="right",fontsize=10)
        ax.set_ylabel("Top minus bottom quartile accuracy (points)",fontsize=11)
        ax.set_title(NAMES[name],fontsize=12)
        ax.grid(axis="y",alpha=0.25)
        ax.tick_params(labelsize=10)
    values = np.load(results/"eval"/"mathvista_confidence_vision_correlation.npy")
    axes[2].hist(values[~np.isnan(values)],bins=30,color="steelblue",edgecolor="white")
    axes[2].axvline(0,color="black",linewidth=0.8)
    axes[2].axvline(np.nanmedian(values),color="firebrick",linestyle="--",label=f"Median {np.nanmedian(values):.2f}")
    axes[2].set_xlabel("Within rollout correlation of confidence and vision",fontsize=11)
    axes[2].set_ylabel("Questions",fontsize=11)
    axes[2].set_title("MathVista, per question mean",fontsize=12)
    axes[2].legend(loc="upper left",fontsize=10,frameon=False)
    axes[2].tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(figures/"signal_quartiles.png",dpi=150)
    plt.close(fig)


def trajectory_figure(cfg,results,figures):
    """Calibrated confidence and visual engagement over the generation of the fixed illustrative rollouts"""
    chosen = load_json(results/"tables"/"signal_statistics.json")["trajectories"]
    arrays,_,tokens = load_arrays(cfg,"mathvista")
    position = {q:i for i,q in enumerate(arrays["questions"])}
    panels = [("agree","visconf_rollout","highest SelfCert and VisConf rollout"),("disagree","selfcert_rollout","highest SelfCert rollout"),("disagree","visconf_rollout","highest VisConf rollout")]
    low,high,epsilon = (cfg["visconf"][k] for k in ("low_percentile","high_percentile","epsilon"))
    fig,axes = plt.subplots(1,3,figsize=(16,4.6),sharey=True)
    for ax,(case,key,label) in zip(axes,panels):
        info = chosen[case]
        q,r = position[info["question"]],info[key]
        rows = arrays["rows"][q]
        confidence = token_confidence(tokens["kl"],tokens["offsets"],rows,low,high,epsilon)[r]
        vision = tokens["vision"][tokens["offsets"][rows[r]]:tokens["offsets"][rows[r]+1]]
        window = max(5,min(25,len(confidence)//4))
        ax.plot(smooth(confidence,window),color="steelblue",label="Calibrated confidence")
        ax.plot(smooth(vision,window),color="firebrick",label="Visual engagement")
        correct = bool(arrays["correct"][q,r])
        ax.set_title(f"{case}: {label}\n{'correct' if correct else 'incorrect'}, question {info['question']}",fontsize=11)
        ax.set_xlabel("Generated token",fontsize=11)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel(f"Value, moving average",fontsize=11)
    axes[0].legend(loc="upper right",fontsize=10,frameon=False)
    fig.tight_layout()
    fig.savefig(figures/"trajectories.png",dpi=150)
    plt.close(fig)


def adaptive_figure(comparison,adaptive,figures):
    """Accuracy against cost for the controllers and fixed budgets plus where the controllers stop"""
    fig,axes = plt.subplots(2,2,figsize=(12,9),gridspec_kw={"width_ratios":[3,2]})
    for row,name in zip(axes,NAMES):
        sc = comparison[(comparison.dataset == name) & (comparison.method == "SC") & (comparison.N >= 4)]
        row[0].plot(sc.cost,sc.accuracy,marker="o",color="black",label="Fixed N self consistency")
        for n,x,y in zip(sc.N,sc.cost,sc.accuracy):
            row[0].annotate(f"N={n}",(x,y),textcoords="offset points",xytext=(6,-12),fontsize=10)
        part = adaptive[adaptive.dataset == name]
        for family,color,marker in (("Agreement","darkorange","*"),("Quality aware","firebrick","D")):
            point = part[part.policy == family].iloc[0]
            row[0].scatter(point.cost,point.accuracy,s=130,color=color,marker=marker,label=family,zorder=3)
        row[0].set_xlabel("Mean generated tokens per question",fontsize=12)
        row[0].set_ylabel("Accuracy (%)",fontsize=12)
        row[0].set_title(NAMES[name],fontsize=12)
        row[0].grid(alpha=0.25)
        row[0].tick_params(labelsize=11)
        families = ["Agreement","Quality aware"]
        bottoms = np.zeros(2)
        for n,color in zip((4,8,16,32),("seagreen","steelblue","darkorange","firebrick")):
            share = np.array([part[part.policy == f].iloc[0][f"stop_{n}"] for f in families])
            row[1].bar(families,share,bottom=bottoms,color=color,label=f"Stop at {n}")
            bottoms += share
        row[1].set_ylabel("Share of question and order runs",fontsize=12)
        row[1].set_title("Stopping point",fontsize=12)
        row[1].tick_params(labelsize=11)
    axes[1,0].legend(loc="lower right",fontsize=10,frameon=False)
    axes[1,1].legend(loc="upper center",bbox_to_anchor=(0.5,-0.12),ncol=2,fontsize=10,frameon=False)
    fig.tight_layout()
    fig.savefig(figures/"adaptive_tradeoff.png",dpi=150)
    plt.close(fig)


def answer_votes(extracted,rows):
    """Answer strings of a set of rollouts with their vote counts"""
    counts = {}
    for r in rows:
        counts[str(extracted[r])] = counts.get(str(extracted[r]),0)+1
    return dict(sorted(counts.items(),key=lambda kv:-kv[1]))


def qualitative(cfg,results):
    """Selection disagreements revisions and premature stops chosen by fixed rules

    rescue is the first evaluation question where self consistency over all 32 rollouts is wrong and the quality score with consensus is right
    harm is the first one where self consistency is right and the quality score with consensus is wrong
    """
    splits = load_json(results/"ids"/"splits.json")
    out = {}
    for name in cfg["datasets"]:
        arrays,orders,tokens = load_arrays(cfg,name)
        raw = repo_path(cfg["paths"]["raw_dir"])/cfg["datasets"][name]["folder"]/"generations.parquet"
        table = pq.read_table(raw,columns=["question_text","extracted_answer","gold_answer"]).to_pydict()
        position = {q:i for i,q in enumerate(arrays["questions"])}
        low,high,epsilon = (cfg["visconf"][k] for k in ("low_percentile","high_percentile","epsilon"))
        found = {}
        for question in splits[name]["evaluation"]:
            q = position[question]
            group,correct = arrays["group"][q],arrays["correct"][q]
            if name == "math":
                scores = arrays["K"][q]
            else:
                confidence,_ = visconf_scores(tokens["kl"],tokens["vision"],tokens["offsets"],arrays["rows"][q],low,high,epsilon)
                scores = arrays["V"][q]*confidence
            sc,rwc = vote_winner(group),rwc_vote(scores,group)[0]
            kind = "rescue" if (not correct[sc] and correct[rwc]) else "harm" if (correct[sc] and not correct[rwc]) else None
            if kind and kind not in found:
                rows = arrays["rows"][q]
                found[kind] = {"question":question,"text":table["question_text"][rows[0]][:400],"gold":table["gold_answer"][rows[0]],"votes":answer_votes(table["extracted_answer"],rows),
                               "self_consistency_answer":table["extracted_answer"][rows[sc]],"quality_consensus_answer":table["extracted_answer"][rows[rwc]],
                               "correct_rollouts":int(correct.sum())}
        premature = load_json(results/"tables"/f"adaptive_premature_{name}.json")
        stops = {}
        for family,case in premature.items():
            rows = arrays["rows"][position[case["question"]]][case["candidates"]]
            stops[family] = {**{k:case[k] for k in ("question","order","stop_at")},"gold":table["gold_answer"][rows[0]],"votes_at_stop":answer_votes(table["extracted_answer"],rows),
                             "votes_over_all_32":answer_votes(table["extracted_answer"],arrays["rows"][position[case["question"]]])}
        out[name] = {"selection":found,"premature_stops":stops}
    live = pd.read_csv(results/"tables"/"live_per_question.csv")
    calls = {}
    for line in (results/"live"/"calls.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        calls[(record["question"],record["strategy"],record["index"])] = record
    revisions = {}
    for kind,mask in (("wrong_to_correct",(~live.initial_correct) & live["Sequential refinement_correct"]),("correct_to_wrong",live.initial_correct & (~live["Sequential refinement_correct"]))):
        if mask.any():
            row = live[mask].iloc[0]
            revisions[kind] = {"question":row.question,"gold":row.gold,"initial_answer":row.initial,"revised_answer":row["Sequential refinement_answer"],
                               "initial_response_end":calls[(row.question,"parallel",0)]["response"][-400:],"revision_response_end":calls[(row.question,"revision",0)]["response"][-400:]}
    out["revision"] = revisions
    save_json(results/"qualitative"/"examples.json",out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["paths"]["results_dir"])
    figures = results/"figures"
    figures.mkdir(parents=True,exist_ok=True)
    comparison = pd.read_csv(results/"tables"/"method_comparison.csv")
    sampling_curves(comparison,figures)
    failure_decomposition(comparison,figures)
    signal_figure(pd.read_csv(results/"tables"/"signal_quartiles.csv"),results,figures)
    trajectory_figure(cfg,results,figures)
    adaptive_figure(comparison,pd.read_csv(results/"tables"/"adaptive_results.csv"),figures)
    if (results/"tables"/"live_per_question.csv").exists():
        qualitative(cfg,results)
    print("wrote figures")


if __name__ == "__main__":
    main()
