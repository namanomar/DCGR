"""Task-clustering study: how well do different clustering strategies recover each user's tasks?"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from common import by_user
from evaluate import context, run_config
from pipeline import Config

MAIN = "qwen3-emb"


def online_cluster(X, tau, recency=0.0):
    cents, counts, lab = [], [], []
    for i, x in enumerate(X):
        if cents:
            sims = np.array([c @ x / (np.linalg.norm(c) + 1e-9) for c in cents])
            if recency:
                last = np.array([max(j for j, l in enumerate(lab) if l == k) for k in range(len(cents))])
                sims = sims + recency * np.exp(-(i - last) / 8.0)
            k = int(sims.argmax())
            if sims[k] >= tau:
                cents[k] = cents[k] + x
                lab.append(k)
                continue
        cents.append(x.copy())
        lab.append(len(cents) - 1)
    return lab


def evaluate_method(users, fn, filtered):
    t, p = [], []
    for u, evs in users.items():
        idx = [i for i, e in enumerate(evs) if (not filtered) or e["task_id"] is not None]
        sub = [evs[i] for i in idx]
        lab = fn(sub)
        for e, l in zip(sub, lab):
            if e["task_id"] is not None:
                t.append(e["task_id"]); p.append(f"{u}-{l}")
    return float(adjusted_rand_score(t, p))


def run():
    cx = context(MAIN)
    emb = cx["emb"]
    ev = cx["ev"]
    res = {}
    for split in ("val", "test"):
        users = by_user([e for e in ev if e["split"] == split])
        Xs = {u: emb.encode([e["text"] for e in evs], is_query=False) for u, evs in users.items()}
        cache = {}

        def mk(method, param):
            def fn(sub):
                key = (sub[0]["user_id"], len(sub), sub[-1]["idx"])
                X = emb.encode([e["text"] for e in sub], is_query=False)
                if method == "online":
                    return online_cluster(X, param)
                if method == "online+recency":
                    return online_cluster(X, param, recency=0.15)
                if method == "agglo":
                    if len(X) < 2:
                        return [0] * len(X)
                    return AgglomerativeClustering(n_clusters=None, distance_threshold=param, metric="cosine", linkage="average").fit_predict(X).tolist()
                if method == "hdbscan":
                    if len(X) < 4:
                        return list(range(len(X)))
                    l = HDBSCAN(min_cluster_size=2, metric="cosine").fit_predict(X)
                    nxt = l.max() + 1
                    out = []
                    for v in l:
                        if v == -1:
                            out.append(nxt); nxt += 1
                        else:
                            out.append(int(v))
                    return out
            return fn
        grid = {"online": [0.5, 0.6, 0.7, 0.8], "online+recency": [0.5, 0.6, 0.7, 0.8], "agglo": [0.3, 0.4, 0.5, 0.6, 0.7], "hdbscan": [0]}
        for m, params in grid.items():
            for filt in (True, False):
                for prm in params:
                    res[f"{split}|{m}|{'taskonly' if filt else 'fullstream'}|{prm}"] = evaluate_method(users, mk(m, prm), filt)
    # choose params on val, report test
    best = {}
    for m in ("online", "online+recency", "agglo", "hdbscan"):
        for mode in ("taskonly", "fullstream"):
            cand = {k.split("|")[3]: v for k, v in res.items() if k.startswith(f"val|{m}|{mode}")}
            bp = max(cand, key=cand.get)
            best[f"{m}|{mode}"] = dict(param=bp, val_ARI=cand[bp], test_ARI=res[f"test|{m}|{mode}|{bp}"])
    # DCGR task assignment (the pipeline)
    for nm, cfg in {"DCGR family-aware (full)": Config(name="a", embed=MAIN), "DCGR without gate (noise enters clusters)": Config(name="b", embed=MAIN, gate=False),
                    "DCGR without notification linking": Config(name="c", embed=MAIN, notif=False)}.items():
        best[nm] = dict(test_ARI=run_config(cfg)["task_ARI"])
    emb.save()
    return dict(grid=res, best=best)
