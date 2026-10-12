import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from collections import Counter, defaultdict
from evaluate import run_config
from pipeline import Config
from embedders import get_embedder
cfg = Config(name="dcgr", embed="qwen3-emb")
r = run_config(cfg, keep_recs=True)
print({k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
for k, v in r["kind"].items(): print(k, {m: round(x, 3) for m, x in v.items()})
errs = defaultdict(list); tk = []
for u, evs, recs in r["_recs"]:
    for e, rc in zip(evs, recs):
        if e["retrieve"] and not (rc["ranked"] and rc["ranked"][0] in e["gold"]):
            errs[e["kind"]].append((e["text"][:70], e["gold"][0], (rc["ranked"] or ["-"])[0], rc["product"], e["product"], rc["pred_type"], rc["q"][:80] if rc["q"] else None))
        if e["task_id"] and e["source"] == "user" and e["etype"] == "query" and rc["task"] is None: tk.append(e["text"])
for k, v in errs.items():
    print("\n==", k, len(v))
    for x in v[:7]: print("  ", x)
get_embedder("qwen3-emb").save()
