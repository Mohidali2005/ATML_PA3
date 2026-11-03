import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import load_json,load_yaml,repo_path
from task1_finetune.utils import CONDITIONS

METHODS = ["full","l2sp","lora4","lora16","oplora"]
NAMES = {"baseline":"Baseline","full_mid":"Full FT (mid)","full":"Full FT","l2sp":"Full FT + L2-SP",
         "lora4":"LoRA-4","lora16":"LoRA-16","oplora":"OPLoRA-16"}
COLORS = {"full":"steelblue","l2sp":"darkorange","lora4":"seagreen","lora16":"firebrick","oplora":"mediumpurple",
          "baseline":"slategray","full_mid":"lightsteelblue"}
# dashing the runs whose curves sit almost on top of another run
LINESTYLES = {"l2sp":"--","oplora":"--"}
PRIMARY = ["gsm8k","hellaswag_norm","boolq","arc_easy_norm"]
# the per example file and column behind each primary metric
SAMPLE_COLUMNS = {"gsm8k":("gsm8k","correct"),"gsm8k_rescued":("gsm8k","rescued"),"hellaswag_norm":("hellaswag","acc_norm"),
                  "boolq":("boolq","acc"),"arc_easy_norm":("arc_easy","acc_norm")}
COMPARISONS = [("full_mid","baseline"),("full","baseline"),("l2sp","baseline"),("lora4","baseline"),
               ("lora16","baseline"),("oplora","baseline"),("full","full_mid"),("lora16","lora4"),
               ("l2sp","full"),("oplora","lora16"),("full","lora16")]


def metric(scores,task,name):
    """Find a metric in the harness output whatever its filter suffix is"""
    key = next(k for k in scores[task] if k.split(",")[0] == name)
    return 100*scores[task][key]


def metric_with_filter(scores,task,name,filter_name):
    """Find a metric of one answer filter in the harness output"""
    return 100*scores[task][f"{name},{filter_name}"]


def boxed_number(text):
    """The number inside the last boxed answer of a response or None when there is none"""
    # dropping text units such as minutes so the number inside is readable while fractions stay unread
    boxed = re.findall(r"\\boxed\{([^{}]*)\}",re.sub(r"\\text\{[^{}]*\}","",text))
    found = re.search(r"-?\d+(?:\.\d+)?",boxed[-1].replace(",","")) if boxed else None
    return float(found.group()) if found else None


def gsm8k_samples(results,condition):
    """GSM8K results per question with failed extractions marked and rescued from the boxed answer"""
    samples = pd.read_json(repo_path(f"{results}/eval/{condition}_gsm8k_samples.jsonl"),lines=True,dtype=False)
    # the flexible filter can keep a latex delimiter such as $$ that holds no number at all
    samples["no_number"] = ~samples["extracted"].astype(str).str.contains(r"\d")
    boxed = samples["response"].map(boxed_number).astype(float)
    target = samples["target"].astype(str).str.replace(",","").astype(float)
    samples["rescued"] = (samples["correct"].astype(bool) | (samples["no_number"] & (boxed == target))).astype(float)
    return samples


def benchmark_table(results):
    """One row per condition with the four benchmarks and the change from the baseline in points"""
    rows = []
    for c in CONDITIONS:
        s = load_json(f"{results}/eval/{c}_scores.json")
        samples = gsm8k_samples(results,c)
        rows.append({"condition":c,
                     "gsm8k":metric_with_filter(s,"gsm8k_cot","exact_match","flexible-extract"),
                     "gsm8k_strict":metric_with_filter(s,"gsm8k_cot","exact_match","strict-match"),
                     "gsm8k_rescued":100*samples["rescued"].mean(),
                     "gsm8k_truncated":100*samples["truncated"].mean(),
                     "gsm8k_no_number":100*samples["no_number"].mean(),
                     "hellaswag_raw":metric(s,"hellaswag","acc"),"hellaswag_norm":metric(s,"hellaswag","acc_norm"),
                     "boolq":metric(s,"boolq","acc"),
                     "arc_easy_raw":metric(s,"arc_easy","acc"),"arc_easy_norm":metric(s,"arc_easy","acc_norm")})
    table = pd.DataFrame(rows)
    for column in list(SAMPLE_COLUMNS):
        table[column+"_delta"] = table[column]-table.loc[0,column]
    table.round(2).to_csv(repo_path(f"{results}/tables/benchmark_table.csv"),index=False)
    print(table.round(2).to_string(index=False))


def correctness(results,condition,task,column):
    """Per question correctness of one condition indexed by question id"""
    path = f"{results}/eval/{condition}_{task}_samples"
    if task == "gsm8k":
        table = gsm8k_samples(results,condition)
    else:
        table = pd.read_csv(repo_path(path+".csv"))
    return table.set_index("doc_id")[column]


def paired_bootstrap(results,seed,resamples=4000):
    """Paired differences in points with 95 percent intervals from resampling questions"""
    rows = []
    for benchmark,(task,column) in SAMPLE_COLUMNS.items():
        scores = pd.DataFrame({c:correctness(results,c,task,column) for c in CONDITIONS})
        assert scores.notna().all().all()
        diffs = np.stack([scores[a].to_numpy()-scores[b].to_numpy() for a,b in COMPARISONS])
        # reseeding per benchmark so every comparison is scored on the same resampled questions
        rng = np.random.default_rng(seed)
        n = diffs.shape[1]
        means = np.stack([diffs[:,rng.integers(0,n,n)].mean(axis=1) for _ in range(resamples)])
        low,high = np.percentile(100*means,[2.5,97.5],axis=0)
        for i,(a,b) in enumerate(COMPARISONS):
            rows.append({"comparison":f"{NAMES[a]} minus {NAMES[b]}","benchmark":benchmark,
                         "difference":100*diffs[i].mean(),"low":low[i],"high":high[i],
                         "excludes_zero":bool(low[i] > 0 or high[i] < 0)})
    table = pd.DataFrame(rows)
    table.round(2).to_csv(repo_path(f"{results}/tables/paired_differences.csv"),index=False)
    table["cell"] = table.apply(lambda r:f"{r['difference']:+.2f} [{r['low']:.2f}, {r['high']:.2f}]",axis=1)
    print(table.pivot(index="comparison",columns="benchmark",values="cell")[list(SAMPLE_COLUMNS)].to_string())


def smooth(values,window=25):
    """Moving average that keeps the first points instead of dropping them"""
    return pd.Series(values).rolling(window,min_periods=1).mean().values


def loss_figure(results):
    """Training loss and learning rate of the five runs"""
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for m in METHODS:
        log = pd.DataFrame(load_json(f"{results}/tables/{m}_training_log.json"))
        style = LINESTYLES.get(m,"-")
        axes[0].plot(log["update"],smooth(log["loss"]),color=COLORS[m],linestyle=style,label=NAMES[m])
        axes[1].plot(log["update"],log["lr"],color=COLORS[m],linestyle=style)
    axes[0].set_xlabel("optimizer update")
    axes[0].set_ylabel("SFT loss (25 update average)")
    axes[0].set_title("(a) Training loss")
    axes[1].set_xlabel("optimizer update")
    axes[1].set_ylabel("learning rate")
    axes[1].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    axes[1].set_title("(b) Learning rate schedule")
    fig.legend(loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(0.5,-0.02))
    fig.tight_layout(rect=(0,0.08,1,1))
    fig.savefig(repo_path(f"{results}/figures/loss_curves.png"),dpi=200)
    plt.close(fig)


def regularization_figure(results):
    """L2-SP penalty over training and the final displacement of the full model runs"""
    log = pd.DataFrame(load_json(f"{results}/tables/l2sp_training_log.json"))
    full = load_json(f"{results}/tables/full_run_summary.json")
    l2sp = load_json(f"{results}/tables/l2sp_run_summary.json")
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(log["update"],smooth(log["loss"]),color="steelblue",label="SFT loss")
    axes[0].plot(log["update"],smooth(log["penalty"]),color="darkorange",label="penalty")
    axes[0].plot(log["update"],smooth(log["total_loss"]),color="firebrick",linestyle="--",label="total objective")
    axes[0].set_xlabel("optimizer update")
    axes[0].set_ylabel("value")
    axes[0].set_title("(a) L2-SP objective terms")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower left",ncol=3,frameon=False,bbox_to_anchor=(0.08,-0.02))
    bars = {"Full FT\nmid":full["displacement"]["mid"],"Full FT\nfinal":full["displacement"]["final"],
            "L2-SP\nfinal":l2sp["displacement"]["final"]}
    axes[1].bar(list(bars),list(bars.values()),color=["lightsteelblue","steelblue","darkorange"])
    axes[1].set_ylabel("distance from starting weights")
    axes[1].set_title("(b) Parameter displacement")
    fig.tight_layout(rect=(0,0.08,1,1))
    fig.savefig(repo_path(f"{results}/figures/regularization.png"),dpi=200)
    plt.close(fig)


def resources_table(results):
    """Trainable parameters time memory and storage of the five runs"""
    rows = []
    for m in METHODS:
        s = load_json(f"{results}/tables/{m}_run_summary.json")
        rows.append({"method":NAMES[m],"trainable_params_m":s["trainable_params"]/1e6,"train_hours":s["train_seconds"]/3600,
                     "peak_memory_gb":s["peak_memory_gb"],"final_checkpoint_mb":s["final_checkpoint_bytes"]/1e6,
                     "resume_checkpoint_mb":s["resume_checkpoint_bytes"]/1e6,"skipped_updates":s["skipped_updates"],
                     "reference_mb":s.get("reference_bytes",0)/1e6,"basis_mb":s.get("basis_bytes",0)/1e6,
                     "setup_seconds":s.get("setup_seconds",s.get("reference_seconds",0)),"export_seconds":s["export_seconds"]})
    table = pd.DataFrame(rows)
    table.round(3).to_csv(repo_path(f"{results}/tables/resources_table.csv"),index=False)
    print(table.round(3).to_string(index=False))


def diagnostics_figure(results):
    """Subspace interference relative update size and activation corrections per layer"""
    table = pd.read_csv(repo_path(f"{results}/tables/diagnostics.csv"))
    panels = [("rho","(a) Subspace interference"),("eta","(b) Relative update size"),
              ("R_math","(c) Activation change on math prompts"),("R_general","(d) Activation change on other prompts")]
    conditions = [c for c in CONDITIONS if c in set(table["condition"])]
    layers = sorted(table["layer"].unique())
    fig,axes = plt.subplots(2,2,figsize=(11,7))
    width = 0.8/len(conditions)
    for ax,(column,title) in zip(axes.ravel(),panels):
        for i,c in enumerate(conditions):
            values = [table[(table["condition"] == c) & (table["layer"] == l)][column].iloc[0] for l in layers]
            ax.bar(np.arange(len(layers))+i*width,np.maximum(values,1e-10),width,color=COLORS[c],label=NAMES[c])
        ax.set_xticks(np.arange(len(layers))+0.4-width/2)
        ax.set_xticklabels([f"layer {l}" for l in layers])
        # only interference needs a log scale since the orthogonal adapter sits near 1e-8
        if column == "rho":
            ax.set_yscale("log")
        ax.set_title(title)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=6,frameon=False,bbox_to_anchor=(0.5,-0.01))
    fig.tight_layout(rect=(0,0.06,1,1))
    fig.savefig(repo_path(f"{results}/figures/diagnostics.png"),dpi=200)
    plt.close(fig)
    pivot = table.pivot_table(index="condition",columns="layer",values=["rho","eta","R_math","R_general"])
    print(pivot.to_string())


def annotation_summary(results):
    """Automatic checks and the manual labels of the 42 annotated responses"""
    out = repo_path(f"{results}/annotation")
    key = pd.read_csv(out/"annotation_key.csv")
    sheet = pd.read_csv(out/"annotation_sheet.csv")
    key = key.merge(sheet[["audit_id","coherent_relevant","math_framing","inappropriate"]],on="audit_id")
    key["condition"] = pd.Categorical(key["condition"],CONDITIONS)
    labelled = key["coherent_relevant"].notna().all()
    summary = key.groupby("condition",observed=True).agg(
        truncated=("truncated","sum"),repeated_trigrams=("repeated_trigrams","mean"),math_markers=("math_markers","mean"),
        valid_json=("valid_json","sum"),template_filled=("template_filled","sum"))
    if labelled:
        summary["coherent_partly"] = key.groupby("condition",observed=True)["coherent_relevant"].apply(lambda s:(s == "partly").sum())
        summary["coherent_no"] = key.groupby("condition",observed=True)["coherent_relevant"].apply(lambda s:(s == "no").sum())
        summary["math_framing"] = key.groupby("condition",observed=True)["math_framing"].apply(lambda s:(s == "yes").sum())
        summary["inappropriate"] = key.groupby("condition",observed=True)["inappropriate"].apply(lambda s:(s == "yes").sum())
    summary.round(3).to_csv(repo_path(f"{results}/tables/annotation_summary.csv"))
    print(summary.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--only",choices=["benchmarks","bootstrap","curves","resources","diagnostics","annotation"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = cfg["paths"]["results_dir"]
    repo_path(f"{results}/figures").mkdir(parents=True,exist_ok=True)
    steps = {"benchmarks":benchmark_table,"resources":resources_table,"diagnostics":diagnostics_figure,"annotation":annotation_summary}
    if args.only in (None,"bootstrap"):
        paired_bootstrap(results,cfg["seed"])
    if args.only in (None,"curves"):
        loss_figure(results)
        regularization_figure(results)
    for name,step in steps.items():
        if args.only in (None,name):
            step(results)


if __name__ == "__main__":
    main()
