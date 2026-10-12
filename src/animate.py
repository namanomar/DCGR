"""Build the animations:
  1. animations/pipeline_demo.html  - self-contained interactive replay of a REAL test-user stream through the pipeline,
                                      switchable between variants (full / minus-one-layer / naive) so you can watch what breaks.
  2. animations/task_clusters.gif   - 2-D projection of one user's events appearing over time, coloured by discovered task.
  3. animations/ablation_race.gif   - metric bars as layers are removed.
"""
from __future__ import annotations
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import run_config, context
from pipeline import Config
from retrieval import rank_metrics

MAIN = "qwen3-emb"
NAIVE = dict(gate=False, tasks=False, constraints=False, notif=False, fresh=False, pfilter=False)

VARIANTS = {
    "full": ("DCGR - all layers", Config(name="full", embed=MAIN)),
    "nogate": ("remove GATE (retrieve on everything)", Config(name="nogate", embed=MAIN, gate=False)),
    "notasks": ("remove TASK CLUSTERING", Config(name="notasks", embed=MAIN, tasks=False)),
    "noconstr": ("remove CONSTRAINT MEMORY", Config(name="noconstr", embed=MAIN, constraints=False)),
    "nonotif": ("remove NOTIFICATION LINKING", Config(name="nonotif", embed=MAIN, notif=False)),
    "naive": ("NAIVE RAG (latest message only)", Config(name="naive", embed=MAIN, qmode="last", **NAIVE)),
}


def pick_user(cx):
    """choose a test user whose stream shows many phenomena (notif seeds, switches, noise)"""
    from common import by_user
    ev = [e for e in cx["ev"] if e["split"] == "test"]
    ub = by_user(ev)
    best, bs = None, -1
    for u, evs in ub.items():
        kinds = {e["kind"] for e in evs}
        sc = len(kinds & {"notif_seed", "notif_ref", "switch_product", "correction", "promo", "otp", "offtopic", "social", "statement", "explicit", "implicit"}) * 10
        sc -= abs(len(evs) - 45)
        if sc > bs:
            best, bs = u, sc
    return best


def trace(cfg, uid, cx, upto=60):
    res = run_config(cfg, keep_recs=True)
    for u, evs, recs in res["_recs"]:
        if u != uid:
            continue
        kb = {d["doc_id"]: d for d in cx["kb"]}
        out = []
        for e, r in list(zip(evs, recs))[:upto]:
            top = (r["ranked"] or [])[:3]
            hit = None
            if e["retrieve"]:
                hit = bool(r["retrieve"] and r["ranked"] and r["ranked"][0] in e["gold"])
            out.append(dict(
                ts=e["ts"][5:16], src=e["source"], text=e["text"], true=e["etype"], kind=e["kind"], pred=r["pred_type"],
                retrieve=bool(r["retrieve"]), want=bool(e["retrieve"]), q=r["q"], task=r["task"], prod=r["product"], psrc=r["prod_src"],
                fam=r.get("family"), top=[dict(id=t, title=kb[t]["title"], ok=t in e["gold"]) for t in top], gold=[kb[g]["title"] for g in e["gold"]], hit=hit))
        return out


def build_html(out_path=ROOT / "animations" / "pipeline_demo.html", upto=60):
    cx = context(MAIN)
    uid = pick_user(cx)
    data = {}
    for k, (label, cfg) in VARIANTS.items():
        data[k] = dict(label=label, events=trace(cfg, uid, cx, upto))
        h = [e["hit"] for e in data[k]["events"] if e["hit"] is not None]
        data[k]["acc"] = float(np.mean(h)) if h else 0
        data[k]["calls"] = sum(e["retrieve"] for e in data[k]["events"])
        print(k, data[k]["acc"], data[k]["calls"], flush=True)
    tpl = (ROOT / "src" / "demo_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/", "const DATA=" + json.dumps(data, ensure_ascii=False) + ";const USER=" + json.dumps(uid) + ";")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print("wrote", out_path)
    return uid


def build_gifs(uid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter, FuncAnimation
    from sklearn.decomposition import PCA
    cx = context(MAIN)
    emb = cx["emb"]
    from common import by_user
    evs = by_user([e for e in cx["ev"] if e["split"] == "test"])[uid][:60]
    X = emb.encode([e["text"] for e in evs], is_query=False)
    P = PCA(2, random_state=0).fit_transform(X)
    res = run_config(VARIANTS["full"][1], keep_recs=True)
    recs = next(r for u, _, r in res["_recs"] if u == uid)[:60]
    palette = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#8ab17d", "#9b5de5", "#f15bb5", "#00bbf9"]
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=90)
    tasks = sorted({r["task"] for r in recs if r["task"] is not None})
    col = {t: palette[i % len(palette)] for i, t in enumerate(tasks)}

    def draw(n):
        ax.clear()
        ax.set_xlim(P[:, 0].min() - .1, P[:, 0].max() + .1); ax.set_ylim(P[:, 1].min() - .1, P[:, 1].max() + .1)
        ax.set_xticks([]); ax.set_yticks([])
        for i in range(n + 1):
            e, r = evs[i], recs[i]
            if r["task"] is not None:
                ax.scatter(*P[i], s=90, c=col[r["task"]], edgecolors="k", linewidths=0.6, zorder=3, marker="o" if e["source"] == "user" else "s")
            else:
                ax.scatter(*P[i], s=45, c="#bbbbbb", marker="x", zorder=2)
        e, r = evs[n], recs[n]
        ax.scatter(*P[n], s=380, facecolors="none", edgecolors="red", linewidths=2, zorder=4)
        ax.set_title(f"event {n+1}/{len(evs)}  [{e['source']}]  gate={r['pred_type']}  task={r['task']}", fontsize=10)
        ax.text(0.01, -0.06, e["text"][:95], transform=ax.transAxes, fontsize=9, va="top")
        ax.text(0.99, 0.99, "circle=user msg  square=notification  grey x = ignored by gate", transform=ax.transAxes, fontsize=7, ha="right", va="top")
    anim = FuncAnimation(fig, draw, frames=len(evs), interval=700)
    anim.save(ROOT / "animations" / "task_clusters.gif", writer=PillowWriter(fps=1.6))
    plt.close(fig)
    print("wrote task_clusters.gif")


if __name__ == "__main__":
    uid = build_html()
    if "--gif" in sys.argv:
        build_gifs(uid)
    from embedders import get_embedder
    get_embedder(MAIN).save()
