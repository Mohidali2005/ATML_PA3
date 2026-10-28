import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common.utils import load_json,load_yaml,repo_path
from task3_ttc.answers import extract_boxed,grade_math,group_math
from task3_ttc.selection import vote_winner


def read_calls(path):
    """Saved calls keyed by question strategy and index"""
    calls = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        calls[(record["question"],record["strategy"],record["index"])] = record
    return calls


def gold_answers(cfg):
    """Reference answer of every MATH question"""
    raw = repo_path(cfg["paths"]["raw_dir"])/cfg["datasets"]["math"]["folder"]/"generations.parquet"
    table = pq.read_table(raw,columns=["sample_id","gold_answer"]).to_pydict()
    return dict(zip(table["sample_id"],table["gold_answer"]))


def strategy_calls(question,calls):
    """The saved calls each strategy pays for, the first parallel response is charged to both parallel sampling and refinement"""
    parallel = [calls[(question,"parallel",i)] for i in range(3)]
    return {"Single longer response":[calls[(question,"single",0)]],"Parallel sampling":parallel,"Sequential refinement":[parallel[0],calls[(question,"revision",0)]]}


def question_outcomes(question,calls,gold):
    """Answers and correctness of every strategy for one question"""
    parallel = [extract_boxed(calls[(question,"parallel",i)]["response"]) for i in range(3)]
    single = extract_boxed(calls[(question,"single",0)]["response"])
    revised = extract_boxed(calls[(question,"revision",0)]["response"])
    labels = group_math(parallel)
    voted = parallel[vote_winner(np.array(labels))]
    answers = {"Single longer response":single,"Parallel sampling":voted,"Sequential refinement":revised}
    return {"initial":parallel[0],"answers":answers,"correct":{k:bool(grade_math(gold,a)) for k,a in answers.items()},"initial_correct":bool(grade_math(gold,parallel[0]))}


def bootstrap_gap(a,b,samples,seed):
    """Mean difference of two per question arrays with a 95 percent interval"""
    gap = np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    index = np.random.default_rng(seed).integers(0,len(gap),size=(samples,len(gap)))
    return float(gap.mean()),*np.percentile(gap[index].mean(axis=1),[2.5,97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--live-dir",default=None)
    ap.add_argument("--output-dir",default=None)
    ap.add_argument("--limit",type=int,default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["paths"]["results_dir"])
    live_dir = repo_path(args.live_dir) if args.live_dir else results/"live"
    out_dir = repo_path(args.output_dir) if args.output_dir else results
    (out_dir/"tables").mkdir(parents=True,exist_ok=True)
    questions = load_json(results/"ids"/"splits.json")["live"][:args.limit]
    calls,golds = read_calls(live_dir/"calls.jsonl"),gold_answers(cfg)
    outcomes = {q:question_outcomes(q,calls,golds[q]) for q in questions}
    summary,per_question = [],[]
    for strategy in ("Single longer response","Parallel sampling","Sequential refinement"):
        used = [strategy_calls(q,calls)[strategy] for q in questions]
        flat = [c for group in used for c in group]
        summary.append({"strategy":strategy,"accuracy":100*np.mean([outcomes[q]["correct"][strategy] for q in questions]),
                        "input_tokens":np.mean([sum(c["input_tokens"] for c in g) for g in used]),
                        "output_tokens":np.mean([sum(c["output_tokens"] for c in g) for g in used]),
                        "seconds":np.mean([sum(c["seconds"] for c in g) for g in used]),
                        "peak_gb":max(c["peak_gb"] for c in flat),
                        "truncation_rate":100*np.mean([c["truncated"] for c in flat]),
                        "extraction_rate":100*np.mean([extract_boxed(c["response"]) is not None for c in flat]),
                        "no_final_answer_rate":100*np.mean([outcomes[q]["answers"][strategy] is None for q in questions])})
    pd.DataFrame(summary).to_csv(out_dir/"tables"/"live_summary.csv",index=False)
    correct = {s:[outcomes[q]["correct"][s] for q in questions] for s in ("Single longer response","Parallel sampling","Sequential refinement")}
    differences = []
    for a,b in (("Parallel sampling","Single longer response"),("Sequential refinement","Single longer response"),("Sequential refinement","Parallel sampling")):
        mean,low,high = bootstrap_gap(correct[a],correct[b],cfg["bootstrap_samples"],cfg["seed"])
        differences.append((a,b,100*mean,100*low,100*high))
    pd.DataFrame(differences,columns=["strategy","baseline","difference","low","high"]).to_csv(out_dir/"tables"/"live_differences.csv",index=False)
    transitions = []
    for strategy in ("Parallel sampling","Sequential refinement"):
        before = np.array([outcomes[q]["initial_correct"] for q in questions])
        after = np.array(correct[strategy])
        changed = [outcomes[q]["answers"][strategy] != outcomes[q]["initial"] for q in questions]
        transitions.append((strategy,int((~before & after).sum()),int((before & ~after).sum()),int((before & after).sum()),int((~before & ~after).sum()),int(sum(changed))))
    pd.DataFrame(transitions,columns=["strategy","wrong_to_correct","correct_to_wrong","both_correct","both_wrong","answer_changed"]).to_csv(out_dir/"tables"/"live_transitions.csv",index=False)
    for q in questions:
        per_question.append({"question":q,"gold":golds[q],"initial":outcomes[q]["initial"],"initial_correct":outcomes[q]["initial_correct"],
                             **{f"{s}_answer":a for s,a in outcomes[q]["answers"].items()},**{f"{s}_correct":c for s,c in outcomes[q]["correct"].items()}})
    pd.DataFrame(per_question).to_csv(out_dir/"tables"/"live_per_question.csv",index=False)
    print("wrote live tables")


if __name__ == "__main__":
    main()
