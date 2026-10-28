import numpy as np
from scipy.stats import rankdata

BASE_METHODS = ["FirstAns","SC","LogP","SelfCert","SelfCert+RWC","Shortest"]
VISION_METHODS = ["Vision","Vision+RWC","VisConf","VisConf+RWC","Tokenwise","Tokenwise+RWC"]
SCORE_OF = {"LogP":"L","SelfCert":"K","SelfCert+RWC":"K","Vision":"V","Vision+RWC":"V","VisConf":"Z","VisConf+RWC":"Z","Tokenwise":"T","Tokenwise+RWC":"T"}


def vote_winner(groups):
    """Position of the earliest candidate in the answer group with the most votes"""
    labels,first,counts = np.unique(groups,return_index=True,return_counts=True)
    best = np.lexsort((first,-counts))[0]
    return int(first[best])


def rwc_weights(scores):
    """Vote weight of every candidate from its descending average rank"""
    n = len(scores)
    return (n-rankdata(-scores,method="average"))/(n-1)


def rwc_vote(scores,groups):
    """Position of the winning candidate and the vote margin of rank weighted consensus"""
    if len(scores) == 1:
        return 0,1.0
    weights = rwc_weights(scores)
    labels,first,inverse = np.unique(groups,return_index=True,return_inverse=True)
    totals = np.round(np.bincount(inverse,weights=weights),9)
    best = np.lexsort((first,-totals))[0]
    ranked = np.sort(totals)[::-1]
    runner_up = ranked[1] if len(ranked) > 1 else 0.0
    return int(first[best]),float((ranked[0]-runner_up)/totals.sum())


def token_confidence(kl,offsets,rows,low,high,epsilon):
    """Calibrated confidence of every token for a set of candidate rollouts

    The percentiles pool the token values of the given candidates only
    """
    segments = [kl[offsets[r]:offsets[r+1]].astype(np.float64) for r in rows]
    q_low,q_high = np.percentile(np.concatenate(segments),[low,high])
    scale = max(q_high-q_low,epsilon)
    return [np.clip((s-q_low)/scale,0,1) for s in segments]


def visconf_scores(kl,vision,offsets,rows,low,high,epsilon):
    """Mean calibrated confidence and its tokenwise fusion with vision for a set of candidate rollouts"""
    confidence = token_confidence(kl,offsets,rows,low,high,epsilon)
    mean_confidence = np.array([c.mean() for c in confidence])
    tokenwise = np.array([(c*vision[offsets[r]:offsets[r+1]]).mean() for c,r in zip(confidence,rows)])
    return mean_confidence,tokenwise


def select_all(groups,tokens,scores,methods):
    """Chosen position of every method among one set of candidates

    scores holds the quality score arrays that exist for the dataset under the keys L K V Z and T
    """
    chosen = {}
    for method in methods:
        if method == "FirstAns":
            chosen[method] = 0
        elif method == "SC":
            chosen[method] = vote_winner(groups)
        elif method == "Shortest":
            chosen[method] = int(np.argmin(tokens))
        elif method.endswith("+RWC"):
            chosen[method] = rwc_vote(scores[SCORE_OF[method]],groups)[0]
        else:
            chosen[method] = int(np.argmax(scores[SCORE_OF[method]]))
    return chosen
