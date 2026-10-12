from __future__ import annotations
import json, re, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


def load_with_retry(fn, tries=8, wait=25):
    """Retry a GPU model-load callable: this box's MIG slice is shared with other tenants
    whose memory usage fluctuates independently, so a transient alloc failure isn't a real bug."""
    for i in range(tries):
        try:
            return fn()
        except RuntimeError:
            if i == tries - 1:
                raise
            try:
                import torch, gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(wait)


def load_kb():
    return [json.loads(l) for l in open(ROOT / "data" / "kb.jsonl", encoding="utf-8")]


def load_events():
    return [json.loads(l) for l in open(ROOT / "data" / "events.jsonl", encoding="utf-8")]


def by_user(events):
    d = {}
    for e in events:
        d.setdefault(e["user_id"], []).append(e)
    for v in d.values():
        v.sort(key=lambda e: e["idx"])
    return d


# ---------- rule lexicon for product mentions (deliberately incomplete: misses ~1/4 of the paraphrases)
LEXICON = {
    "term": [r"(?<![-\w])term\b(?!-)", r"\bterm (plan|policy|insurance|cover)\b"],
    "ulip": [r"\bulips?\b", r"\bunit[- ]linked\b"],
    "endow": [r"\bendowment\b", r"\bsavings (plan|policy)\b", r"\bassured savings\b"],
    "pension": [r"\bpension\b", r"\bannuity\b", r"\bretirement (policy|plan)\b"],
    "mf": [r"\bmutual fund\b", r"\bmf sip\b", r"\bsip folio\b", r"\bfolio\b"],
}
_LEX = {p: [re.compile(x, re.I) for x in xs] for p, xs in LEXICON.items()}


def lexicon_products(text: str) -> list[tuple[int, str]]:
    """Return [(position, product)] sorted by first appearance."""
    hits = []
    for p, rs in _LEX.items():
        pos = [m.start() for r in rs for m in [r.search(text)] if m]
        if pos:
            hits.append((min(pos), p))
    return sorted(hits)
