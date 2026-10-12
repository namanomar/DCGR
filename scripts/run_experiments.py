"""Run every experiment of the study. Results -> results/*.json (resumable: finished stages are skipped).

usage:  python scripts/run_experiments.py [stage ...]
stages: embed search context ablation noise cluster llm bootstrap
"""
import sys, json, time
from pathlib import Path
from dataclasses import asdict, replace
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from evaluate import run_config, context
from pipeline import Config
from embedders import get_embedder

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
MAIN = "qwen3-emb"
NAIVE = dict(gate=False, tasks=False, constraints=False, notif=False, fresh=False, pfilter=False)


def slim(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def save(name, obj):
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print(f"[saved] {name}", flush=True)


def done(name):
    return (RES / f"{name}.json").exists()


def stage_embed():
    """E1 - which embedding model backs the dense side + the lightweight heads?"""
    out = {}
    for emb in ["tfidf", "minilm", "bge-small", "qwen3-emb", "e5-large", "gte-large", "qwen3-emb-4b"]:
        for retr in ["bm25", "dense", "hybrid_rrf", "hybrid_w"]:
            if emb == "tfidf" and retr in ("hybrid_rrf", "hybrid_w"):
                pass
            cfg = Config(name=f"{emb}/{retr}", embed=emb, retr=retr)
            t = time.time()
            out[f"{emb}|{retr}"] = slim(run_config(cfg))
            print(emb, retr, f"R@1={out[f'{emb}|{retr}']['R@1']:.3f} nDCG={out[f'{emb}|{retr}']['nDCG@10']:.3f} ({time.time()-t:.0f}s)", flush=True)
        get_embedder(emb).save()
        get_embedder(emb).free()  # bound GPU memory across the 7-model sweep (qwen3-emb-4b alone is ~8GB fp16)
    save("embed", out)


def stage_search():
    """E2 - search strategies with the main embedder (+ cross-encoder rerank)."""
    out = {}
    for retr in ["bm25", "dense", "hybrid_rrf", "hybrid_w"]:
        for rr in [False, True]:
            cfg = Config(name=f"{retr}{'+rerank' if rr else ''}", embed=MAIN, retr=retr, rerank=rr)
            out[cfg.name] = slim(run_config(cfg))
            print(cfg.name, f"R@1={out[cfg.name]['R@1']:.3f} nDCG={out[cfg.name]['nDCG@10']:.3f} ms={out[cfg.name]['ms_p50']:.1f}", flush=True)
    # same, but naive context (latest message only) to show what search alone achieves
    for retr in ["bm25", "dense", "hybrid_rrf"]:
        cfg = Config(name=f"naive-{retr}", embed=MAIN, retr=retr, qmode="last", **NAIVE)
        out[cfg.name] = slim(run_config(cfg))
        print(cfg.name, f"R@1={out[cfg.name]['R@1']:.3f}", flush=True)
    get_embedder(MAIN).save()
    save("search", out)


def stage_context():
    """E3 - how is the retrieval query built from the conversation?"""
    out = {}
    base = dict(embed=MAIN, retr="hybrid_rrf")
    variants = {
        "latest message only (naive)": Config(name="last", qmode="last", **NAIVE, **base),
        "last-4 events concat (naive)": Config(name="concat", qmode="concat", **NAIVE, **base),
        "last-12 user history (naive)": Config(name="history", qmode="history", **NAIVE, **base),
        "last-4 concat + gate": Config(name="concat+gate", qmode="concat", gate=True, tasks=False, constraints=False, notif=False, fresh=False, pfilter=False, **base),
        "structured task state (DCGR)": Config(name="dcgr", **base),
        "oracle context (gold product + referent)": Config(name="oracle", qmode="oracle", **base),
    }
    for k, cfg in variants.items():
        out[k] = slim(run_config(cfg))
        print(k, f"R@1={out[k]['R@1']:.3f} MRR={out[k]['MRR@10']:.3f}", flush=True)
    get_embedder(MAIN).save()
    save("context", out)


def stage_ablation():
    """E4 - remove one layer at a time."""
    base = Config(name="DCGR (full)", embed=MAIN, retr="hybrid_rrf")
    variants = {
        "DCGR (full)": base,
        "- gate (retrieve on every event)": replace(base, gate=False),
        "- task clustering (single rolling ctx)": replace(base, tasks=False),
        "- constraint memory (no product slot/profile)": replace(base, constraints=False),
        "- notification linking": replace(base, notif=False),
        "- freshness filter": replace(base, fresh=False),
        "- product metadata filter": replace(base, pfilter=False),
        "- hybrid (BM25 only)": replace(base, retr="bm25"),
        "- hybrid (dense only)": replace(base, retr="dense"),
        "+ cross-encoder rerank": replace(base, rerank=True),
        "- gate - tasks - constraints - notif": replace(base, gate=False, tasks=False, constraints=False, notif=False),
        "naive baseline (all layers removed)": Config(name="naive", embed=MAIN, retr="hybrid_rrf", qmode="last", **NAIVE),
    }
    out = {}
    for k, cfg in variants.items():
        out[k] = slim(run_config(cfg))
        print(k, f"R@1={out[k]['R@1']:.3f} nDCG={out[k]['nDCG@10']:.3f} calls={out[k]['calls_per_100_events']:.1f}", flush=True)
    get_embedder(MAIN).save()
    save("ablation", out)


def stage_noise():
    """E5 - interruption stress: add 0 / 0.5 / 1 / 2 / 3 extra noise events per event."""
    out = {}
    cfgs = {
        "naive (latest only)": Config(name="naive", embed=MAIN, retr="hybrid_rrf", qmode="last", **NAIVE),
        "naive (last-4 concat)": Config(name="concat", embed=MAIN, retr="hybrid_rrf", qmode="concat", **NAIVE),
        "DCGR no-gate": Config(name="dcgr-nogate", embed=MAIN, gate=False),
        "DCGR no-notif-link": Config(name="dcgr-nonotif", embed=MAIN, notif=False),
        "DCGR (full)": Config(name="dcgr", embed=MAIN),
    }
    for rate in [0, 0.5, 1, 2, 3]:
        for k, cfg in cfgs.items():
            r = slim(run_config(cfg, noise=rate, seed=11))
            out[f"{rate}|{k}"] = r
            print(f"noise={rate}", k, f"R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} calls={r['retrieval_calls']}", flush=True)
    get_embedder(MAIN).save()
    save("noise", out)


def stage_llm():
    """E6 - small local LLMs (Qwen3 0.6B / 1.7B): rewrite baseline vs. selective resolver (DCGR+LLM)."""
    from llm import LocalLLM
    out = {}
    for mname in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct"]:
        llm = LocalLLM(mname, batch=4)
        tag = mname.split("/")[-1]
        for k, cfg in {
            f"LLM rewrite of last-6 events (no gate) [{tag}]": Config(name="rw", embed=MAIN, qmode="llm_rewrite", llm=True, **NAIVE),
            f"LLM rewrite + gate [{tag}]": Config(name="rw+gate", embed=MAIN, qmode="llm_rewrite", llm=True, gate=True, tasks=False, constraints=False, notif=False, fresh=False, pfilter=False),
            f"DCGR + selective LLM resolver [{tag}]": Config(name="dcgr-llm", embed=MAIN, llm=True),
        }.items():
            t = time.time()
            r = slim(run_config(cfg, llm=llm))
            r["llm_calls"] = llm.calls
            r["llm_seconds"] = llm.seconds
            out[k] = r
            print(k, f"R@1={r['R@1']:.3f} llm_calls_new={llm.calls} ({time.time()-t:.0f}s)", flush=True)
            llm.calls = 0
        llm.free()
        save("llm", out)
        get_embedder(MAIN).save()


def stage_cluster():
    from cluster_study import run
    save("cluster", run())


def stage_bootstrap():
    """95% CI over users (bootstrap) for the headline systems."""
    cfgs = {
        "naive": Config(name="naive", embed=MAIN, qmode="last", **NAIVE),
        "naive-concat": Config(name="concat", embed=MAIN, qmode="concat", **NAIVE),
        "DCGR": Config(name="dcgr", embed=MAIN),
    }
    out = {}
    rng = np.random.default_rng(0)
    for k, cfg in cfgs.items():
        r = run_config(cfg, keep_recs=True)
        per_user = []
        from retrieval import rank_metrics
        for u, evs, recs in r["_recs"]:
            m = [rank_metrics(rc["ranked"] if rc["retrieve"] and rc["ranked"] else [], set(e["gold"])) for e, rc in zip(evs, recs) if e["retrieve"]]
            if m:
                per_user.append({kk: np.mean([x[kk] for x in m]) for kk in ("R@1", "R@5", "MRR@10", "nDCG@10")} | {"n": len(m)})
        res = {}
        for kk in ("R@1", "R@5", "MRR@10", "nDCG@10"):
            vals = np.array([p[kk] for p in per_user]); w = np.array([p["n"] for p in per_user])
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, len(vals), len(vals))
                boots.append(np.average(vals[idx], weights=w[idx]))
            res[kk] = dict(mean=float(np.average(vals, weights=w)), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)))
        out[k] = res
        print(k, {kk: round(v["mean"], 3) for kk, v in res.items()}, flush=True)
    save("bootstrap", out)


STAGES = dict(embed=stage_embed, search=stage_search, context=stage_context, ablation=stage_ablation, noise=stage_noise,
              llm=stage_llm, cluster=stage_cluster, bootstrap=stage_bootstrap)

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith('--')] or list(STAGES)
    for s in want:
        if done(s) and "--force" not in sys.argv:
            print("skip", s)
            continue
        STAGES[s]()
