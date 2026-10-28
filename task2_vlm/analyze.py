import argparse
import csv
import textwrap

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import load_json,load_yaml,read_jsonl,repo_path
from task2_vlm.evaluation import CONDITIONS
from task2_vlm.utils import fetch_image

COLORS = {"correct":"seagreen","swap":"firebrick","mismatch":"darkorange","none":"steelblue"}
LABELS = {"correct":"Correct image","swap":"Complementary swap","mismatch":"Mismatched image","none":"No image"}
SUITE_NAMES = {"grounding_dev":"Grounding","vqa_dev":"Natural VQA","grounding_test":"Grounding","vqa_test":"Natural VQA"}


def write_csv(path,header,rows):
    """Write a table of rows with a header"""
    with open(path,"w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def smooth(values,window):
    """Moving average that keeps the length of the input"""
    values = np.asarray(values,dtype=float)
    kernel = np.ones(window)/window
    padded = np.concatenate([np.full(window-1,values[0]),values])
    return np.convolve(padded,kernel,mode="valid")


def learning_curves(history,dev,ends,figures):
    """Training loss and development metrics across the three stages"""
    fig,axes = plt.subplots(2,3,figsize=(17,9))
    updates = [h["update"] for h in history]
    axes[0,0].plot(updates,smooth([h["loss"] for h in history],25),color="steelblue",linewidth=1.6)
    axes[0,0].set_title("(a) Training loss (25 update average)",fontsize=13)
    axes[0,0].set_ylabel("Loss per example")
    points = dev
    axes[0,1].plot([d["update"] for d in points],[d["caption_loss"] for d in points],color="seagreen",marker="o",linewidth=1.6)
    axes[0,1].set_title("(b) Development caption loss",fontsize=13)
    axes[0,1].set_ylabel("Loss per caption")
    for ax,suite,title in ((axes[0,2],"grounding_dev","(c) Grounding answer NLL"),(axes[1,0],"vqa_dev","(d) Natural VQA answer NLL")):
        for c in ("correct","swap","none"):
            ax.plot([d["update"] for d in points],[d[suite]["nll"][c] for d in points],color=COLORS[c],marker="o",linewidth=1.6,label=LABELS[c])
        ax.set_title(title,fontsize=13)
        ax.set_ylabel("NLL per answer")
    for suite,color in (("grounding_dev","steelblue"),("vqa_dev","darkorange")):
        axes[1,1].plot([d["update"] for d in points],[d[suite]["delta_nll"] for d in points],color=color,marker="o",linewidth=1.6,label=SUITE_NAMES[suite])
    axes[1,1].axhline(0,color="gray",linewidth=0.8)
    axes[1,1].set_title("(e) Image swap penalty",fontsize=13)
    axes[1,1].set_ylabel("NLL(partner) - NLL(correct)")
    ends_dev = [d for d in dev if "accuracy" in d["grounding_dev"]]
    for suite,style in (("grounding_dev","-"),("vqa_dev","--")):
        for c in ("correct","swap"):
            axes[1,2].plot([d["update"] for d in ends_dev],[d[suite]["accuracy"][c] for d in ends_dev],color=COLORS[c],linestyle=style,marker="o",linewidth=1.6,
                           label=f"{SUITE_NAMES[suite]}, {LABELS[c].lower()}")
    axes[1,2].set_title("(f) Development accuracy at stage ends",fontsize=13)
    axes[1,2].set_ylabel("Accuracy")
    for ax in axes.flat:
        for e in ends[:-1]:
            ax.axvline(e,color="gray",linestyle=":",linewidth=1)
        ax.set_xlabel("Update")
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.25)
    for ax in (axes[0,2],axes[1,0],axes[1,1],axes[1,2]):
        ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=2,fontsize=10,frameon=False)
    for ax in axes.flat:
        ax.set_title(ax.get_title(),fontsize=13,pad=22)
        low,high = ax.get_xlim()
        for n,(a,b) in enumerate(zip([0]+list(ends[:-1]),ends)):
            # skipping stage labels that fall outside the panel
            if low < (a+b)/2 < high:
                ax.text((a+b)/2,1.0,f"Stage {n+1}",ha="center",va="bottom",fontsize=10,color="dimgray",transform=ax.get_xaxis_transform())
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.55)
    fig.savefig(figures/"learning_curves.png",dpi=150)
    plt.close(fig)


def final_tables(scores,tables):
    """Flat tables of the final results"""
    rows,types,advantage,outcomes = [],[],[],[]
    for suite,conds in scores["scores"].items():
        for c in CONDITIONS:
            s,i = conds[c],scores["intervals"][suite][c]
            rows.append([suite,c,s["accuracy"],i["accuracy"]["low"],i["accuracy"]["high"],s["pair_accuracy"],i["pair_accuracy"]["low"],i["pair_accuracy"]["high"],
                         s["identical_rate"],s["empty_rate"],s["nll"],i["nll"]["low"],i["nll"]["high"],s["seconds_per_answer"]])
            types += [[suite,c,t,v] for t,v in s["by_type"].items()]
        for c,m in scores["correct_image_advantage"][suite].items():
            for metric,v in m.items():
                advantage.append([suite,f"correct minus {c}" if metric != "nll" else f"{c} minus correct",metric,v["mean"],v["low"],v["high"]])
        outcomes.append([suite]+[scores["swap_outcomes"][suite][k] for k in ("original_answer","partner_answer","other")])
    write_csv(tables/"final_results.csv",["suite","condition","accuracy","accuracy_low","accuracy_high","pair_accuracy","pair_low","pair_high",
                                          "identical_rate","empty_rate","nll","nll_low","nll_high","seconds_per_answer"],rows)
    write_csv(tables/"final_by_type.csv",["suite","condition","type","accuracy"],types)
    write_csv(tables/"correct_image_advantage.csv",["suite","comparison","metric","mean","low","high"],advantage)
    write_csv(tables/"swap_outcomes.csv",["suite","original_answer","partner_answer","other"],outcomes)


def resources(summary,cache,scores,tables):
    """Compact resource table by stage"""
    rows = []
    for name,s in summary["stages"].items():
        rows.append([name,s["connector_params"],s["adapter_params"],s["train_seconds"],s["eval_seconds"],s["peak_gb"]])
    rows.append(["feature_caching",cache["encoder_params"],0,cache["total_seconds"],0,cache["peak_memory_gb"]])
    rows.append(["final_evaluation",0,0,0,scores["eval_seconds"],scores["peak_memory_gb"]])
    write_csv(tables/"resources.csv",["stage","connector_params","adapter_params","train_seconds","eval_seconds","peak_gb"],rows)


def pick_pairs(results,suite):
    """First pair in file order for each outcome type

    grounded means both answers are right and both swapped answers follow the partner image
    image ignored means swapping the image changes neither answer
    failure means a wrong answer with the correct image and no change after the swap
    """
    correct = read_jsonl(f"{results}/eval/{suite}_correct.jsonl")
    swap = read_jsonl(f"{results}/eval/{suite}_swap.jsonl")
    chosen = {}
    for p in range(len(correct)//2):
        c,s = correct[2*p:2*p+2],swap[2*p:2*p+2]
        kind = None
        if all(r["exact"] for r in c) and all(r["partner_match"] for r in s):
            kind = "grounded"
        elif all(a["prediction"] == b["prediction"] for a,b in zip(c,s)):
            kind = "image ignored"
        elif not all(r["exact"] for r in c) and not any(r["partner_match"] for r in s):
            kind = "failure"
        if kind and kind not in chosen:
            chosen[kind] = (c,s)
    return chosen


def qualitative(results,suite,figures):
    """Figure of complementary pairs with the shared question and the answers under both images"""
    chosen = pick_pairs(results,suite)
    fig,axes = plt.subplots(len(chosen),2,figsize=(11,4.6*len(chosen)),squeeze=False)
    for row,(kind,(c,s)) in zip(axes,chosen.items()):
        for i,ax in enumerate(row):
            ax.imshow(fetch_image(c[i]["image"]))
            ax.axis("off")
            text = (f"{kind}: {textwrap.fill(c[i]['question'],40)}\nreference: {c[i]['reference']}\n"
                    f"answer on this image: {c[i]['prediction']}\nanswer on the partner image: {s[i]['prediction']}")
            ax.set_title(text,fontsize=11,loc="left")
    fig.tight_layout()
    fig.savefig(figures/f"qualitative_{suite}.png",dpi=130)
    plt.close(fig)
    return {k:[r["pair_id"] for r in v[0][:1]] for k,v in chosen.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--results-dir")
    ap.add_argument("--skip-images",action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = args.results_dir or cfg["paths"]["results_dir"]
    figures = repo_path(results)/"figures"
    tables = repo_path(results)/"tables"
    figures.mkdir(parents=True,exist_ok=True)
    summary = load_json(f"{results}/tables/run_summary.json")
    scores = load_json(f"{results}/eval/final_scores.json")
    cache = load_json(f"{results}/tables/feature_cache.json")
    ends = list(np.cumsum([s["updates"] for s in load_json(f"{results}/tables/config_used.json")["stages"]]))

    learning_curves(load_json(f"{results}/tables/training_log.json"),load_json(f"{results}/tables/dev_log.json"),ends,figures)
    final_tables(scores,tables)
    resources(summary,cache,scores,tables)
    dev = load_json(f"{results}/tables/dev_log.json")
    captions = [[d["update"],c["image"],c["reference"],c["generated"]] for d in dev if "captions" in d for c in d["captions"][:8]]
    write_csv(tables/"dev_captions.csv",["update","image","reference","generated"],captions)
    if not args.skip_images:
        picked = {s:qualitative(results,s,figures) for s in ("grounding_test","vqa_test")}
        print("qualitative pairs",picked)
    print("wrote figures and tables")


if __name__ == "__main__":
    main()
