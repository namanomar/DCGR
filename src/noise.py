"""Interruption stress-test: inject extra noise events into a user's stream (labelled, synthetic)."""
from __future__ import annotations
import random
from datetime import datetime, timedelta
import gen_data as g


def inject(evs: list[dict], rate: float, seed: int = 0, user=None) -> list[dict]:
    """After each event, with probability `rate` (can exceed 1 -> several) insert a noise event.
    Noise mix: small talk, off-topic, promos (keyword-overlapping), OTP, transactional notifications of OTHER families."""
    if rate <= 0:
        return evs
    rng = random.Random(seed)
    out = []
    n = 0
    for e in evs:
        out.append(e)
        k = int(rate) + (1 if rng.random() < rate - int(rate) else 0)
        for _ in range(k):
            r = rng.random()
            if r < 0.2:
                src, et, txt, kind = "user", "social", rng.choice(g.SOCIAL), "social"
            elif r < 0.4:
                src, et, txt, kind = "user", "offtopic", rng.choice(g.OFFTOPIC), "offtopic"
            elif r < 0.75:
                src, et, txt, kind = rng.choice(["sms", "email", "app"]), "notif_noise", rng.choice(g.PROMOS), "promo"
            elif r < 0.85:
                src, et, txt, kind = "sms", "notif_noise", g.OTP.format(otp=rng.randint(100000, 999999)), "otp"
            else:
                nt = rng.choice(list(g.NOTIF_TEXT))
                prods = [p["product"] for p in (user["policies"] if user else [{"product": "term"}])]
                cand = [p for p in prods if p in g.NOTIF_ASPECT_PRODUCTS[nt]]
                if not cand:
                    continue
                p = rng.choice(cand)
                pol = next((x for x in user["policies"] if x["product"] == p), {"premium": 12000, "last4": "0000"}) if user else {"premium": 12000, "last4": "0000"}
                txt = g.NOTIF_TEXT[nt].format(amt=f"{pol['premium']:,}", pname=g.PRODUCT_NAME[p], last4=pol["last4"], ref=rng.randint(10000, 99999), date="15 Jul", pct="+1.2")
                src, et, kind = "sms", "notif_transactional", "notif_unengaged"
            n += 1
            d = dict(e)
            d.update(source=src, etype=et, text=txt, kind=kind, task_id=None, retrieve=False, gold=[], aspect=None, product=None,
                     dependent=False, family=None, notif_type=nt if kind == "notif_unengaged" else None, event_id=f"{e['event_id']}-n{n}")
            out.append(d)
    for i, e in enumerate(out):
        e = dict(e)
        e["idx"] = i
        out[i] = e
    return out
