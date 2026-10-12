"""Generate all report figures from data/ and results/*.json  ->  figures/*.png"""
import sys, json
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RES = ROOT / "results"
ACC, GREY, WARN, GOOD = "#1f6feb", "#8c96a3", "#d1495b", "#2a9d8f"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "figure.dpi": 140,
                     "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True})


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()
    print("fig", name)


def fig_dataset():
    ev = [json.loads(l) for l in open(ROOT / "data" / "events.jsonl", encoding="utf-8")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    c = Counter(e["kind"] for e in ev)
    order = ["implicit", "explicit", "notif_ref", "switch_product", "correction", "statement", "product_update", "social", "offtopic", "promo", "otp", "system", "notif_seed", "notif_unengaged"]
    vals = [c.get(k, 0) for k in order]
    colors = [ACC if k in ("implicit", "explicit", "notif_ref", "switch_product", "correction") else GOOD if k in ("statement", "product_update", "notif_seed") else GREY for k in order]
    ax[0].barh(order[::-1], vals[::-1], color=colors[::-1])
    ax[0].set_title("Events by kind"); ax[0].set_xlabel("count")
    per = Counter(e["user_id"] for e in ev)
    ax[1].hist(list(per.values()), bins=18, color=ACC)
    ax[1].set_title("Events per user"); ax[1].set_xlabel("events"); ax[1].set_ylabel("users")
    sp = Counter((e["split"], e["retrieve"]) for e in ev)
    labels = ["train", "val", "test"]
    ax[2].bar(labels, [sp[(s, True)] for s in labels], color=ACC, label="needs retrieval")
    ax[2].bar(labels, [sp[(s, False)] for s in labels], bottom=[sp[(s, True)] for s in labels], color=GREY, label="no retrieval needed")
    ax[2].set_title("Split by user (no leakage)"); ax[2].legend(frameon=False, fontsize=8)
    savefig("fig_dataset.png")


def fig_arch():
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 4.6); ax.axis("off")

    def box(x, y, w, h, t, sub="", fc="#eaf1fd", ec=ACC):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h * .64, t, ha="center", va="center", fontweight="bold", fontsize=9.5)
        if sub:
            ax.text(x + w / 2, y + h * .27, sub, ha="center", va="center", fontsize=7.6, color="#445")

    def arr(x1, y1, x2, y2, c="#333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=c, lw=1.3))
    box(.1, 1.7, 1.9, 1.3, "Event stream", "user msgs · SMS · app\nemail · OTP · promos", "#f2f3f5", GREY)
    box(2.5, 1.7, 1.9, 1.3, "1 Gate", "social / off-topic / question\nstatement / notification", "#eaf1fd")
    box(4.9, 1.7, 1.9, 1.3, "2 Task clusters", "per-user online clustering\n(family + recency)", "#eaf1fd")
    box(7.3, 1.7, 1.9, 1.3, "3 Constraint slots", "product · profile · notif\nstated once, reused", "#eaf1fd")
    box(9.7, 1.7, 2.6, 1.3, "4 Query builder", "standalone query from task state\n+ referenced notification", "#eaf1fd")
    box(7.3, 0.1, 2.3, 1.1, "5 Search", "BM25 + dense (RRF)\nfreshness + product filters", "#e8f6f3", GOOD)
    box(10.0, 0.1, 2.3, 1.1, "6 (opt.) Rerank", "cross-encoder top-20", "#e8f6f3", GOOD)
    box(2.5, 3.5, 4.3, 0.9, "Local small LLM (Qwen3 0.6B / 1.7B)", "only for ambiguous / referential events", "#fdf1e7", "#d98324")
    box(.1, 0.1, 6.7, 1.1, "Quarantine: social / off-topic / OTP / promos -> no retrieval, no memory pollution", "", "#fbeaea", WARN)
    for a, b in [(2.0, 2.5), (4.4, 4.9), (6.8, 7.3), (9.2, 9.7)]:
        arr(a, 2.35, b, 2.35)
    arr(11.0, 1.7, 9.0, 1.2); arr(9.6, .65, 10.0, .65)
    arr(3.5, 1.7, 3.5, 1.2, WARN)
    arr(5.0, 3.5, 5.5, 3.0, "#d98324")
    ax.set_title("Dynamic Context-Gated Retrieval (DCGR) - on-premises pipeline", loc="left")
    savefig("fig_arch.png")


def bars(ax, labels, vals, colors=None, fmt="{:.2f}", horizontal=False):
    if horizontal:
        b = ax.barh(labels, vals, color=colors)
        for r, v in zip(b, vals):
            ax.text(v + .005, r.get_y() + r.get_height() / 2, fmt.format(v), va="center", fontsize=8)
    else:
        b = ax.bar(labels, vals, color=colors)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + .005, fmt.format(v), ha="center", fontsize=8)


EMB_LABEL = {
    "tfidf": "TF-IDF", "minilm": "MiniLM-L6 (22M)", "bge-small": "BGE-small (33M)",
    "qwen3-emb": "Qwen3-Emb (600M)", "e5-large": "E5-large-v2 (335M)",
    "gte-large": "GTE-large (335M)", "qwen3-emb-4b": "Qwen3-Emb (4B)",
}


def fig_embed():
    d = load("embed")
    if not d:
        return
    rets = ["bm25", "dense", "hybrid_rrf", "hybrid_w"]
    embs = [e for e in EMB_LABEL if any(f"{e}|{r}" in d for r in rets)]
    n = len(embs)
    fig, ax = plt.subplots(1, 2, figsize=(11, 1.2 + 0.55 * n))
    for a, met in zip(ax, ["R@1", "nDCG@10"]):
        M = np.array([[d[f"{e}|{r}"][met] for r in rets] for e in embs])
        im = a.imshow(M, cmap="Blues", vmin=M.min() - .05, vmax=M.max() + .02)
        a.set_xticks(range(4)); a.set_xticklabels(["BM25", "dense", "hybrid RRF", "hybrid weighted"], fontsize=8, rotation=20, ha="right")
        a.set_yticks(range(n)); a.set_yticklabels([EMB_LABEL[e] for e in embs], fontsize=8)
        a.grid(False)
        for i in range(n):
            for j in range(4):
                a.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9, color="white" if M[i, j] > (M.max() + M.min()) / 2 + .03 else "black")
        a.set_title(f"{met}  (DCGR pipeline, embedder x search)")
    savefig("fig_embed.png")


def fig_search():
    d = load("search")
    if not d:
        return
    names = ["bm25", "dense", "hybrid_rrf", "hybrid_w", "bm25+rerank", "dense+rerank", "hybrid_rrf+rerank", "hybrid_w+rerank"]
    names = [n for n in names if n in d]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    cols = [GOOD if "rerank" in n else ACC for n in names]
    bars(ax[0], names, [d[n]["R@1"] for n in names], cols, "{:.3f}", True)
    ax[0].set_title("R@1 by search strategy (DCGR context)"); ax[0].set_xlim(.5, .95)
    bars(ax[1], names, [d[n]["nDCG@10"] for n in names], cols, "{:.3f}", True)
    ax[1].set_title("nDCG@10"); ax[1].set_xlim(.6, 1.0)
    savefig("fig_search.png")


def fig_context():
    d = load("context")
    if not d:
        return
    names = list(d)
    cols = [GREY if "naive" in n else GOOD if "oracle" in n else ACC for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    bars(ax[0], names, [d[n]["R@1"] for n in names], cols, "{:.3f}", True)
    ax[0].set_title("R@1: how is the query built?"); ax[0].set_xlim(0, 1)
    ax[1].barh(names, [d[n]["calls_per_100_events"] for n in names], color=cols)
    ax[1].set_title("retrieval calls per 100 events"); ax[1].set_yticklabels([])
    savefig("fig_context.png")


def fig_ablation():
    d = load("ablation")
    if not d:
        return
    names = list(d)
    full = d["DCGR (full)"]["R@1"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    cols = [ACC if n == "DCGR (full)" else GOOD if n.startswith("+") else WARN if "naive" in n else GREY for n in names]
    bars(ax[0], names[::-1], [d[n]["R@1"] for n in names][::-1], cols[::-1], "{:.3f}", True)
    ax[0].axvline(full, color=ACC, ls="--", lw=1)
    ax[0].set_xlim(.3, .92); ax[0].set_title("R@1 when one layer is removed")
    ax[1].barh(names[::-1], [d[n]["calls_per_100_events"] for n in names][::-1], color=cols[::-1])
    ax[1].set_yticklabels([]); ax[1].set_title("retrieval calls / 100 events")
    savefig("fig_ablation.png")


def fig_noise():
    d = load("noise")
    if not d:
        return
    rates = sorted({float(k.split("|")[0]) for k in d})
    systems = list(dict.fromkeys(k.split("|")[1] for k in d))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    cols = {s: c for s, c in zip(systems, [GREY, "#5b6572", WARN, "#d98324", ACC])}
    for s in systems:
        ys = [d[f"{r:g}|{s}"]["R@1"] for r in rates]
        ax[0].plot(rates, ys, marker="o", label=s, color=cols[s], lw=2.2 if s == "DCGR (full)" else 1.4)
        ax[1].plot(rates, [d[f"{r:g}|{s}"]["retrieval_calls"] for r in rates], marker="o", color=cols[s], lw=2.2 if s == "DCGR (full)" else 1.4)
    ax[0].set_xlabel("injected noise events per original event"); ax[0].set_ylabel("R@1"); ax[0].set_title("Robustness to interruptions"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel("injected noise events per original event"); ax[1].set_ylabel("retrieval calls"); ax[1].set_title("Search calls triggered")
    savefig("fig_noise.png")


def fig_cluster():
    d = load("cluster")
    if not d:
        return
    b = d["best"]
    names = list(b)
    vals = [b[n]["test_ARI"] for n in names]
    cols = [ACC if n.startswith("DCGR family") else WARN if "without" in n else GREY for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    bars(ax, names[::-1], vals[::-1], cols[::-1], "{:.3f}", True)
    ax.set_xlim(0, 1.05); ax.set_title("Task recovery (adjusted Rand index, held-out users)")
    savefig("fig_cluster.png")


def fig_kind():
    a = load("ablation")
    if not a:
        return
    full, naive = a["DCGR (full)"]["kind"], a["naive baseline (all layers removed)"]["kind"]
    kinds = ["implicit", "explicit", "notif_ref", "switch_product", "correction"]
    kinds = [k for k in kinds if k in full]
    x = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - .2, [naive[k]["R@1"] for k in kinds], .4, color=GREY, label="naive RAG")
    ax.bar(x + .2, [full[k]["R@1"] for k in kinds], .4, color=ACC, label="DCGR")
    ax.set_xticks(x); ax.set_xticklabels([f"{k}\n(n={full[k]['n']})" for k in kinds], fontsize=8)
    ax.set_ylabel("R@1"); ax.set_title("Where context helps: R@1 by question type"); ax.legend(frameon=False)
    savefig("fig_kind.png")


def fig_llm():
    d = load("llm")
    if not d:
        return
    names = list(d)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 1.2 + 0.3 * len(names)))
    cols = [WARN if "rewrite" in n else ACC for n in names]
    bars(ax[0], names[::-1], [d[n]["R@1"] for n in names][::-1], cols[::-1], "{:.3f}", True)
    ax[0].set_xlim(.3, .95); ax[0].set_title("R@1 with small local LLMs")
    ax[1].barh(names[::-1], [d[n].get("llm_calls", 0) for n in names][::-1], color=cols[::-1])
    ax[1].set_yticklabels([]); ax[1].set_title("LLM calls issued")
    savefig("fig_llm.png")


if __name__ == "__main__":
    fig_dataset(); fig_arch(); fig_embed(); fig_search(); fig_context(); fig_ablation(); fig_noise(); fig_cluster(); fig_kind(); fig_llm()
