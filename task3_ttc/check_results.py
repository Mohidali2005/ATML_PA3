import argparse

import numpy as np
import pandas as pd

from common.utils import load_json,load_yaml,repo_path
from task3_ttc.evaluate_offline import load_arrays
from task3_ttc.selection import rwc_vote,vote_winner


def check_selection_rules():
    """Rank weighted consensus with equal scores must reduce to plain voting and one candidate returns itself"""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2,33))
        groups = rng.integers(-1,5,size=n)
        assert rwc_vote(np.ones(n),groups)[0] == vote_winner(groups)
    assert rwc_vote(np.array([0.3]),np.array([2]))[0] == 0 and vote_winner(np.array([2])) == 0


def check_dataset(cfg,name,results):
    """Consistency of the saved evaluation tables with each other and with the raw arrays"""
    table = pd.read_csv(results/"eval"/f"{name}_per_question.csv")
    first = table[table.N == 1].pivot(index="question",columns="method",values="accuracy")
    assert np.allclose(first.sub(first["FirstAns"],axis=0).abs().max().max(),0),"methods differ at one candidate"
    for method,part in table.groupby("method"):
        assert (part.accuracy <= part.oracle+1e-9).all(),f"{method} above oracle coverage"
    oracle = table[table.method == "SC"].pivot(index="question",columns="N",values="oracle")
    assert (np.diff(oracle.to_numpy(),axis=1) >= -1e-9).all(),"oracle coverage falls with more candidates"
    arrays,orders,_ = load_arrays(cfg,name)
    position = {q:i for i,q in enumerate(arrays["questions"])}
    first_cost = table[(table.method == "FirstAns") & (table.N == 8)].set_index("question").cost
    for question in list(first_cost.index)[:50]:
        q = position[question]
        assert np.isclose(first_cost[question],np.mean([arrays["tokens"][q,orders[q,o,0]] for o in range(orders.shape[1])]))
    sc_cost = table[(table.method == "SC") & (table.N == 4)].set_index("question").cost
    for question in list(sc_cost.index)[:50]:
        q = position[question]
        assert np.isclose(sc_cost[question],np.mean([arrays["tokens"][q,orders[q,o,:4]].sum() for o in range(orders.shape[1])]))
    comparison = pd.read_csv(results/"tables"/"method_comparison.csv")
    adaptive = pd.read_csv(results/"tables"/"adaptive_results.csv")
    for n in (4,8,16,32):
        offline = comparison[(comparison.dataset == name) & (comparison.method == "SC") & (comparison.N == n)].iloc[0]
        online = adaptive[(adaptive.dataset == name) & (adaptive.policy == f"Fixed {n}")].iloc[0]
        assert np.isclose(offline.accuracy,online.accuracy) and np.isclose(offline.cost,online.cost),f"adaptive fixed {n} differs from offline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["paths"]["results_dir"])
    check_selection_rules()
    for name in cfg["datasets"]:
        check_dataset(cfg,name,results)
    statistics = load_json(results/"tables"/"signal_statistics.json")
    assert statistics["mathvista"]["covariance_identity_max_error"] < 1e-5
    print("all checks passed")


if __name__ == "__main__":
    main()
