"""Search layer: BM25, TF-IDF cosine, dense (cosine), hybrid fusion (RRF / weighted), optional
cross-encoder rerank, and metadata filters (freshness, product)."""
from __future__ import annotations
import math, re
from collections import Counter
import numpy as np

STOP = set("a an the of to in on for and or is are was be do does did i my me we you your it this that these those with at by from as if can could would should will what how when where which who whom there their then than so into about any have has had not no yes please".split())
_tok = re.compile(r"[a-z0-9]+")


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(s: str) -> list[str]:
    return [stem(w) for w in _tok.findall(s.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avg = sum(len(t) for t in self.toks) / self.N
        df = Counter(w for t in self.toks for w in set(t))
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(t) for t in self.toks]
        self.len = np.array([len(t) for t in self.toks])

    def scores(self, q: str) -> np.ndarray:
        qt = tokenize(q)
        s = np.zeros(self.N, dtype=np.float32)
        for w in set(qt):
            if w not in self.idf:
                continue
            idf = self.idf[w]
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def rrf(rank_lists: list[np.ndarray], k: int = 60, weights: list[float] | None = None) -> np.ndarray:
    """Reciprocal-rank fusion of score arrays -> fused scores."""
    n = len(rank_lists[0])
    out = np.zeros(n, dtype=np.float32)
    for j, sc in enumerate(rank_lists):
        order = np.argsort(-sc, kind="stable")
        r = np.empty(n, dtype=np.int32)
        r[order] = np.arange(n)
        out += (weights[j] if weights else 1.0) / (k + r + 1)
    return out


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


class Searcher:
    """Index of the KB under one embedder + BM25. `search` returns ranked doc indices."""

    def __init__(self, kb: list[dict], embedder, rerank_model: str | None = None):
        self.kb = kb
        self.ids = [d["doc_id"] for d in kb]
        self.docs = [f"{d['title']}. {d['text']}" for d in kb]
        self.bm25 = BM25(self.docs)
        self.emb = embedder
        self.D = embedder.encode(self.docs, is_query=False)
        self.status = np.array([d["status"] == "active" for d in kb])
        self.product = np.array([d["product"] for d in kb])
        self._ce = None
        self.rerank_model = rerank_model

    def _mask(self, product: str | None, fresh: bool) -> np.ndarray:
        m = np.ones(len(self.kb), dtype=bool)
        if fresh:
            m &= self.status
        if product:
            m &= (self.product == product) | (self.product == "all")
        return m

    def raw_scores(self, q: str, mode: str) -> np.ndarray:
        if mode == "bm25":
            return self.bm25.scores(q)
        if mode in ("dense", "tfidf"):
            return self.D @ self.emb.encode([q], is_query=True)[0]
        if mode == "hybrid_rrf":
            return rrf([self.bm25.scores(q), self.D @ self.emb.encode([q], True)[0]])
        if mode == "hybrid_w":
            return 0.4 * minmax(self.bm25.scores(q)) + 0.6 * minmax(self.D @ self.emb.encode([q], True)[0])
        raise ValueError(mode)

    def search(self, q: str, mode="hybrid_rrf", product: str | None = None, fresh: bool = False, rerank: bool = False, topk_rerank: int = 20) -> list[int]:
        s = self.raw_scores(q, mode).astype(np.float64)
        m = self._mask(product, fresh)
        s = np.where(m, s, -1e9)
        order = np.argsort(-s, kind="stable")
        if rerank:
            order = self._rerank(q, order, m, topk_rerank)
        return [int(i) for i in order]

    def _rerank(self, q, order, m, k):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            import torch
            self._ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda" if torch.cuda.is_available() else "cpu", max_length=256)
        head = [int(i) for i in order[:k]]
        sc = self._ce.predict([(q, self.docs[i]) for i in head], show_progress_bar=False, batch_size=32)
        head = [h for _, h in sorted(zip(-np.asarray(sc), head))]
        return np.array(head + [int(i) for i in order[k:]])


# ----------------------------------------------------------------------- metrics
def rank_metrics(ranked_ids: list[str], gold: set[str], ks=(1, 3, 5, 10)) -> dict:
    out = {}
    pos = [i for i, d in enumerate(ranked_ids) if d in gold]
    first = pos[0] if pos else None
    for k in ks:
        out[f"R@{k}"] = float(len(set(ranked_ids[:k]) & gold) / max(1, len(gold)))
    out["MRR@10"] = 1.0 / (first + 1) if first is not None and first < 10 else 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i in pos if i < 10)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    out["nDCG@10"] = dcg / idcg if idcg else 0.0
    return out
