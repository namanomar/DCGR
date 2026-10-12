"""Experiment harness: run a Config over a split and compute retrieval / gate / task / efficiency metrics."""
from __future__ import annotations
import json, time
from collections import defaultdict
import numpy as np
from sklearn.metrics import adjusted_rand_score

from common import load_kb, load_events, by_user, ROOT
from embedders import get_embedder
from retrieval import Searcher, rank_metrics
from pipeline import Pipeline, Models, Config
from kb import PRODUCT_NAME

_CTX = {}


def context(embed: str):
    if embed in _CTX:
        return _CTX[embed]
    kb = load_kb()
    ev = load_events()
    users = json.load(open(ROOT / "data" / "users.json", encoding="utf-8"))
    corpus = [f"{d['title']}. {d['text']}" for d in kb] + [e["text"] for e in ev]
    emb = get_embedder(embed, corpus)
    searcher = Searcher(kb, emb)
    train = [e for e in ev if e["split"] == "train"]
    models = Models(emb, train, users, kbD=searcher.D)
    _CTX[embed] = dict(kb=kb, ev=ev, users={u["user_id"]: u for u in users}, emb=emb, searcher=searcher, models=models)
    return _CTX[embed]


def run_config(cfg: Config, split="test", llm=None, max_users=None, keep_recs=False, noise=0.0, seed=0):
    cx = context(cfg.embed)
    ev = [e for e in cx["ev"] if e["split"] == split]
    uidx = by_user(ev)
    uids = sorted(uidx)[:max_users] if max_users else sorted(uidx)
    pipe = Pipeline(cx["kb"], cx["searcher"], cx["models"], cfg, llm=llm)
    allrecs = []
    if noise > 0:
        from noise import inject
        for u in uids:
            uidx[u] = inject(uidx[u], noise, seed=seed, user=cx["users"][u])
    if cfg.retr != "bm25" and not (cfg.qmode == "llm_rewrite" and llm is None):
        # pass 1 (dry run): collect every query string, embed them in ONE batch, so pass 2 hits the cache
        pipe.collect = []
        for u in uids:
            recs = pipe.run_user(uidx[u], cx["users"][u])
            if cfg.qmode == "oracle":
                oracle_fix(pipe, uidx[u], recs, cfg)
        qs = sorted(set(pipe.collect))
        pipe.collect = None
        cx["emb"].encode(qs, is_query=True)
        cx["emb"].save()
    t0 = time.perf_counter()
    for u in uids:
        recs = pipe.run_user(uidx[u], cx["users"][u])
        if cfg.qmode == "oracle":
            recs = oracle_fix(pipe, uidx[u], recs, cfg)
        allrecs.append((u, uidx[u], recs))
    wall = time.perf_counter() - t0
    res = metrics(allrecs, cfg, cx)
    res["wall_s"] = wall
    if keep_recs:
        res["_recs"] = allrecs
    return res


def oracle_fix(pipe, evs, recs, cfg):
    for e, r in zip(evs, recs):
        if e["retrieve"]:
            # oracle context: perfect resolution of product + referents. Self-contained messages are used as-is;
            # elliptic / deictic ones are replaced by the canonical question of the gold aspect.
            if e["kind"] in ("switch_product", "correction", "notif_ref"):
                from gen_data import Q
                q = Q[e["aspect"]][0]
            else:
                q = e["text"]
            r["q"], r["retrieve"] = q, True
            r["ranked"] = pipe._search(q, e["product"] if cfg.pfilter else None)
        else:
            r["retrieve"] = False
            r["ranked"] = None
    return recs


def metrics(allrecs, cfg, cx):
    agg = defaultdict(list)
    bykind = defaultdict(lambda: defaultdict(list))
    tp = fp = fn = tn = 0
    etype_ok = []
    notif_ok = []
    true_t, pred_t = [], []
    prod_ok = []
    ms = []
    n_events = n_ret = 0
    for u, evs, recs in allrecs:
        for e, r in zip(evs, recs):
            n_events += 1
            ms.append(r.get("ms", 0))
            if e["source"] == "user":
                etype_ok.append(r["pred_type"] == e["etype"])
            else:
                notif_ok.append((r["pred_type"] == "notif_transactional") == (e["etype"] == "notif_transactional"))
            want = bool(e["retrieve"])
            did = bool(r["retrieve"])
            n_ret += did
            tp += want and did
            fp += (not want) and did
            fn += want and not did
            tn += (not want) and (not did)
            if want:
                ranked = r["ranked"] if r["retrieve"] and r["ranked"] else []
                m = rank_metrics(ranked, set(e["gold"]))
                for k, v in m.items():
                    agg[k].append(v)
                    bykind[e["kind"]][k].append(v)
                if e["family"] in ("premium", "claim", "withdrawal", "tax", "charges"):
                    prod_ok.append(r["product"] == e["product"])
            if e["task_id"] is not None and (e["source"] == "user" or e["kind"] == "notif_seed"):
                true_t.append(e["task_id"])
                pred_t.append(f"{u}-{r['task']}" if r.get("task") is not None else f"{u}-none-{e['idx']}")
    out = {k: float(np.mean(v)) for k, v in agg.items()}
    out["n_queries"] = len(agg["R@1"])
    out["kind"] = {k: {m: float(np.mean(v)) for m, v in d.items()} | {"n": len(d["R@1"])} for k, d in bykind.items()}
    prec = tp / max(1, tp + fp)
    rec_ = tp / max(1, tp + fn)
    out["gate_precision"], out["gate_recall"] = prec, rec_
    out["gate_f1"] = 2 * prec * rec_ / max(1e-9, prec + rec_)
    out["retrieval_calls"] = int(n_ret)
    out["wasted_calls"] = int(fp)
    out["calls_per_100_events"] = 100 * n_ret / max(1, n_events)
    out["etype_acc"] = float(np.mean(etype_ok)) if etype_ok else None
    out["notif_acc"] = float(np.mean(notif_ok)) if notif_ok else None
    out["task_ARI"] = float(adjusted_rand_score(true_t, pred_t)) if true_t else None
    out["product_acc"] = float(np.mean(prod_ok)) if prod_ok else None
    out["ms_p50"] = float(np.percentile(ms, 50))
    out["ms_p95"] = float(np.percentile(ms, 95))
    out["n_events"] = n_events
    return out


def fmt(res, keys=("R@1", "R@5", "MRR@10", "nDCG@10", "gate_f1", "calls_per_100_events", "task_ARI", "product_acc", "ms_p50")):
    return " ".join(f"{k}={res[k]:.3f}" if isinstance(res.get(k), float) else f"{k}={res.get(k)}" for k in keys)


if __name__ == "__main__":
    import sys
    embed = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
    for name, cfg in {
        "naive-last/bm25": Config(name="naive", embed=embed, retr="bm25", gate=False, tasks=False, constraints=False, notif=False, fresh=False, pfilter=False, qmode="last"),
        "DCGR-full": Config(embed=embed),
        "oracle": Config(embed=embed, qmode="oracle"),
    }.items():
        r = run_config(cfg)
        print(f"{name:20s}", fmt(r))
    get_embedder(embed).save()
