"""DCGR - Dynamic Context-Gated Retrieval for continuous, interrupted, multi-user conversations.

Layers (each can be switched off for ablations):
  L1 gate        classify every incoming event (social / off-topic / query / statement / notification kinds)
                 and decide whether external retrieval is worth running at all.
  L2 tasks       per-user online task clustering: an event joins an existing task cluster, starts a new one,
                 or (when ambiguous) goes to a small local LLM resolver.
  L3 constraints persistent slots per task (product), stated early and reused implicitly later,
                 plus the user profile (CRM) as fallback.
  L4 notif       transactional notifications (premium due, claim update ...) are buffered as candidate task
                 seeds; promos / OTPs are quarantined so they cannot contaminate queries.
  L5 query       build a context-aware standalone query from the task state (structured), or baselines.
  L6 search      BM25 / dense / hybrid, metadata filters (freshness, product), optional cross-encoder rerank.
"""
from __future__ import annotations
import re, time
from dataclasses import dataclass, field, asdict
import numpy as np
from sklearn.linear_model import LogisticRegression

from kb import PRODUCT_NAME, FAMILIES, ASPECT_FAMILY
from common import lexicon_products
from retrieval import Searcher
from llm import rewrite_prompt, resolver_prompt, REWRITE_SYS

PRODUCT_DEP_FAMILIES = {"premium", "claim", "withdrawal", "tax", "charges"}
NOTIF_FAMILY = {"premium_due": "premium", "claim_update": "claim", "nav_alert": "charges", "sip_failed": "sip", "kyc_pending": "service", "maturity_reminder": "claim"}
USER_TYPES = ["social", "offtopic", "query", "statement"]
ELLIPTIC_KINDS = {"switch_product", "correction"}
DEICTIC_KINDS = {"notif_ref"}
MENTION_KINDS = {"explicit", "switch_product", "correction", "product_update"}


@dataclass
class Config:
    name: str = "DCGR-full"
    embed: str = "qwen3-emb"
    retr: str = "hybrid_rrf"        # bm25 | tfidf | dense | hybrid_rrf | hybrid_w
    rerank: bool = False
    gate: bool = True
    tasks: bool = True
    constraints: bool = True
    notif: bool = True
    fresh: bool = True
    pfilter: bool = True
    qmode: str = "structured"       # last | concat | history | llm_rewrite | structured | oracle
    llm: bool = False               # allow local LLM resolver for ambiguous events
    gate_tau: float = 0.0           # LLM fallback if gate confidence below this (only when llm=True)
    fam_tau: float = 0.45           # family-confidence threshold for task assignment


# ---------------------------------------------------------------------------------------------
class Models:
    """Trained light-weight heads on top of frozen embeddings (trained on TRAIN users only)."""

    def __init__(self, emb, train_events, users, kbD=None):
        self.emb = emb
        self.kbD = kbD
        self.profile = {u["user_id"]: u for u in users}
        tr = [e for e in train_events]
        X = self.feat(tr)
        usr = [i for i, e in enumerate(tr) if e["source"] == "user"]
        ntf = [i for i, e in enumerate(tr) if e["source"] != "user"]
        self.c_user = LogisticRegression(max_iter=2000, C=10).fit(X[usr], [tr[i]["etype"] for i in usr])
        self.c_notif = LogisticRegression(max_iter=2000, C=10).fit(X[ntf], [tr[i]["etype"] for i in ntf])
        # family (query / statement / transactional notif)
        fi = [i for i, e in enumerate(tr) if (e["source"] == "user" and e["etype"] in ("query", "statement") and e.get("family"))
              or (e["etype"] == "notif_transactional" and e.get("notif_type"))]
        fl = [tr[i]["family"] if tr[i]["source"] == "user" else NOTIF_FAMILY[tr[i]["notif_type"]] for i in fi]
        self.c_family = LogisticRegression(max_iter=3000, C=10).fit(X[fi], fl)
        # dependency: standalone / elliptic / deictic
        qi = [i for i in usr if tr[i]["etype"] == "query"]
        ql = ["elliptic" if tr[i]["kind"] in ELLIPTIC_KINDS else "deictic" if tr[i]["kind"] in DEICTIC_KINDS else "standalone" for i in qi]
        self.c_dep = LogisticRegression(max_iter=2000, C=10, class_weight="balanced").fit(X[qi], ql)
        # does message mention a product / which one
        mi = [i for i in usr if tr[i]["etype"] in ("query", "statement")]
        self.c_mention = LogisticRegression(max_iter=2000, C=10).fit(X[mi], [tr[i]["kind"] in MENTION_KINDS for i in mi])
        pi = [i for i in mi if tr[i]["kind"] in MENTION_KINDS]
        self.c_prod = LogisticRegression(max_iter=2000, C=10).fit(X[pi], [tr[i]["product"] for i in pi])
        # notification type
        ni = [i for i in ntf if tr[i]["etype"] == "notif_transactional" and tr[i].get("notif_type")]
        self.c_ntype = LogisticRegression(max_iter=2000, C=10).fit(X[ni], [tr[i]["notif_type"] for i in ni])

    def feat(self, evs):
        v = self.emb.encode([e["text"] for e in evs], is_query=False)
        if self.kbD is not None:  # relevance-to-KB signal: how close is this message to ANY knowledge-base article?
            ms = (v @ self.kbD.T).max(1, keepdims=True)
            v = np.hstack([v, 4.0 * ms])
        return v

    def user_type(self, x):
        p = self.c_user.predict_proba(x[None])[0]
        return self.c_user.classes_[p.argmax()], float(p.max())

    def notif_type(self, x):
        p = self.c_notif.predict_proba(x[None])[0]
        return self.c_notif.classes_[p.argmax()], float(p.max())

    def family(self, x):
        p = self.c_family.predict_proba(x[None])[0]
        return self.c_family.classes_[p.argmax()], float(p.max())

    def dep(self, x):
        p = self.c_dep.predict_proba(x[None])[0]
        return self.c_dep.classes_[p.argmax()], float(p.max())

    def product_from_text(self, text, x):
        hits = lexicon_products(text)
        if hits:
            return hits[0][1], "lexicon"
        if self.c_mention.predict_proba(x[None])[0][1] > 0.6:
            p = self.c_prod.predict_proba(x[None])[0]
            if p.max() > 0.6:
                return self.c_prod.classes_[p.argmax()], "classifier"
        return None, None

    def ntype(self, x):
        return self.c_ntype.predict(x[None])[0]


# ---------------------------------------------------------------------------------------------
def family_products(kb):
    fp = {}
    for d in kb:
        if d["kind"] == "core" and d["product"] != "all":
            fp.setdefault(d["family"], set()).add(d["product"])
    return fp


class Pipeline:
    def __init__(self, kb, searcher: Searcher, models: Models, cfg: Config, llm=None):
        self.kb, self.S, self.M, self.cfg, self.llm = kb, searcher, models, cfg, llm
        self.fp = family_products(kb)
        self.kb_ids = searcher.ids

    def llm_query(self, prompt, system):
        out = self.llm.gen(system, [prompt])[0]
        return out if out else prompt.split("Customer's latest message:")[-1][:200]

    # ---------------- helpers
    def _valid_product(self, prod, fam):
        allowed = self.fp.get(fam)
        if not allowed:
            return None  # product-agnostic family (docs are 'all')
        if prod in allowed:
            return prod
        return next(iter(allowed)) if len(allowed) == 1 else None

    collect = None  # when a list: dry-run mode that only records queries (so they can be embedded in one batch)

    def _search(self, q, prod):
        c = self.cfg
        if self.collect is not None:
            self.collect.append(q)
            return []
        order = self.S.search(q, mode=c.retr, product=prod if c.pfilter else None, fresh=c.fresh, rerank=c.rerank)
        return [self.kb_ids[i] for i in order]

    # ---------------- main loop over one user's stream
    def run_user(self, evs, user):
        c, M = self.cfg, self.M
        X = M.feat(evs)
        tasks, nbuf, recs, hist = [], [], [], []
        next_tid = 0
        profile_prods = [p["product"] for p in user["policies"]]
        profile_unique = profile_prods[0] if len(profile_prods) == 1 else None
        roll = dict(product=None, last_q=None)  # single rolling context (when tasks=False)
        for i, e in enumerate(evs):
            t0 = time.perf_counter()
            x = X[i]
            rec = dict(event_id=e["event_id"], idx=e["idx"], pred_type=None, retrieve=False, q=None, ranked=None, task=None,
                       product=None, prod_src=None, llm_calls=0, family=None, dep=None, ts=e["ts"], text=e["text"], source=e["source"])
            text = e["text"]
            is_user = e["source"] == "user"
            # ---------------- L1 gate
            if c.gate:
                if is_user:
                    ptype, conf = M.user_type(x)
                else:
                    ptype, conf = M.notif_type(x)
            else:
                ptype, conf = ("query" if is_user else "notif_transactional"), 1.0
            rec["pred_type"] = ptype
            # ---------------- notifications
            if not is_user:
                if ptype == "notif_transactional" and c.notif:
                    nt = M.ntype(x)
                    hits = lexicon_products(text)
                    nbuf.append(dict(idx=i, family=NOTIF_FAMILY[nt], product=hits[0][1] if hits else None, used=False, ntype=nt, rec=rec, text=text))
                    rec["family"] = NOTIF_FAMILY[nt]
                    rec["product"] = hits[0][1] if hits else None
                hist.append(e)
                rec["ms"] = (time.perf_counter() - t0) * 1000
                recs.append(rec)
                continue
            # ---------------- user events
            if ptype in ("social", "offtopic"):
                hist.append(e)
                rec["ms"] = (time.perf_counter() - t0) * 1000
                recs.append(rec)
                continue
            fam, fconf = M.family(x)
            dep, dconf = M.dep(x)
            if ptype == "statement":
                dep = "standalone"
            rec["family"], rec["dep"] = fam, dep
            mprod, msrc = M.product_from_text(text, x)
            # deictic ("what happens if I miss THIS?"): resolve the referent among buffered service notifications
            nref = None
            if dep == "deictic" and c.notif:
                pf = dict(zip(M.c_family.classes_, M.c_family.predict_proba(x[None])[0]))
                cands = [(pf.get(n["family"], 0) + 0.15 * (1 - (i - n["idx"]) / 40), n) for n in nbuf if i - n["idx"] < 40 and not n["used"]]
                if cands:
                    nref = max(cands, key=lambda z: z[0])[1]
                    fam = nref["family"]
                    rec["family"] = fam
            # ---------------- L2 task assignment
            task = None
            if c.tasks:
                cand = [t for t in tasks if t["family"] == fam]
                if cand and (dep == "standalone" and fconf >= c.fam_tau):
                    task = cand[-1]
                elif dep == "deictic":
                    task = cand[-1] if (cand and nref is not None) else None  # a notification-seeded task is created below
                elif dep != "standalone" or fconf < c.fam_tau:
                    # ambiguous / elliptic: attach to the most recent task (recency prior)
                    task = next((t for t in reversed(tasks) if t["last_q"]), tasks[-1] if tasks else None)
                if task is None:
                    task = dict(id=next_tid, family=fam, product=None, last_q=None, n=0, first=i)
                    next_tid += 1
                    tasks.append(task)
                    if c.notif:  # seed from a pending transactional notification of the same family
                        for nb in ([nref] if nref is not None else reversed(nbuf)):
                            if nb["family"] == fam and not nb["used"] and i - nb["idx"] < 40:
                                nb["used"] = True
                                task["product"] = task["product"] or nb["product"]
                                task["seed"] = nb
                                nb["rec"]["task"] = task["id"]
                                break
                tasks.remove(task)
                tasks.append(task)  # most recent last
                rec["task"] = task["id"]
            # ---------------- L3 constraints (product slot)
            prod, psrc = None, None
            if c.constraints:
                if mprod:
                    prod, psrc = mprod, msrc
                elif c.tasks and task and task["product"]:
                    prod, psrc = task["product"], "task"
                elif (not c.tasks) and roll["product"]:
                    prod, psrc = roll["product"], "roll"
                elif c.notif:
                    nb = next((n for n in reversed(nbuf) if n["family"] == fam and n["product"] and i - n["idx"] < 40), None)
                    if nb:
                        prod, psrc = nb["product"], "notif"
                if prod is None and profile_unique:
                    prod, psrc = profile_unique, "profile"
                # when the user states a product in a standalone-aspect query: remember it
                if mprod and task is not None:
                    task["product"] = mprod
                if mprod:
                    roll["product"] = mprod
            vprod = self._valid_product(prod, fam)
            rec["product"], rec["prod_src"] = vprod, psrc
            if ptype == "statement":
                hist.append(e)
                rec["ms"] = (time.perf_counter() - t0) * 1000
                recs.append(rec)
                continue
            # ---------------- L5 query construction
            q = self._build_query(c.qmode, e, evs, i, hist, task, roll, dep, vprod, fam, mprod, nbuf)
            if c.qmode == "structured" and nref is not None:
                q = f"{q} {evs[nref['idx']]['text']}"[:400]
            if c.qmode == "llm_rewrite" and self.llm is not None:
                q = self.llm_query(rewrite_prompt(hist[-6:], text), REWRITE_SYS)
                rec["llm_calls"] += 1
            elif c.llm and self.llm is not None and c.qmode == "structured" and (dep != "standalone" or fconf < c.fam_tau):
                ctx_tasks = tasks if c.tasks else [dict(family=fam, product=roll["product"], last_q=roll["last_q"])]
                q = self.llm_query(resolver_prompt(ctx_tasks, [n for n in nbuf if i - n["idx"] < 40 and n["family"] == fam or not c.gate][-2:] if c.notif else [], text), REWRITE_SYS)
                rec["llm_calls"] += 1
                lp = lexicon_products(q)
                if lp and c.constraints:
                    vprod = self._valid_product(lp[0][1], fam)
                    rec["product"] = vprod
            rec["q"] = q
            # ---------------- L6 search
            rec["retrieve"] = True
            rec["ranked"] = self._search(q, vprod if c.constraints else None)
            # memory update
            if dep == "standalone":
                if task is not None:
                    task["last_q"] = text
                roll["last_q"] = text
            # (elliptic / deictic turns keep the original standalone question as the carried aspect)
            hist.append(e)
            rec["ms"] = (time.perf_counter() - t0) * 1000
            recs.append(rec)
        return recs

    # ---------------------------------------------------------------------------------------
    def _build_query(self, mode, e, evs, i, hist, task, roll, dep, vprod, fam, mprod, nbuf):
        text = e["text"]
        if mode == "last":
            return text
        if mode == "concat":  # last 4 events of ANY source (what a naive chat-RAG would do)
            ctx = [h["text"] for h in hist[-4:]]
            return " ".join(ctx + [text])
        if mode == "history":  # all prior user messages in the stream window (full-history baseline)
            ctx = [h["text"] for h in hist[-12:] if h["source"] == "user"]
            return " ".join(ctx + [text])
        if mode == "oracle":
            return text  # oracle handled by evaluator (gold aspect/product)
        if mode in ("structured", "llm_rewrite"):
            prod_name = PRODUCT_NAME.get(vprod, "") if (vprod and fam in PRODUCT_DEP_FAMILIES and not self.cfg.pfilter) else ""
            base = text
            if dep == "elliptic":
                lastq = (task["last_q"] if (self.cfg.tasks and task) else roll["last_q"]) or ""
                base = lastq or text
            return f"{base} {prod_name}".strip()
        return text
