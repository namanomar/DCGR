"""Tiny local LLM wrapper (Qwen3-0.6B up to ~4B via transformers, 4-bit on GPU). Nothing leaves the machine.
Every generation is cached on disk by prompt hash so experiments are reproducible and cheap to re-run."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
from common import load_with_retry

CACHE = Path(__file__).resolve().parent.parent / "data" / "llm_cache"
CACHE.mkdir(parents=True, exist_ok=True)

REWRITE_SYS = ("You rewrite the customer's latest chat message into ONE self-contained search query for an insurance and mutual-fund "
               "knowledge base. Resolve references like 'this', 'it', 'the other one' using ONLY the relevant earlier context. "
               "Ignore unrelated promos, OTPs and small talk. Mention the product (term plan, ULIP, savings plan, pension plan, mutual fund) if it is known. "
               "Output only the query, nothing else.")


class LocalLLM:
    def __init__(self, name="Qwen/Qwen3-1.7B", batch=4, max_new=40):
        self.name, self.batch, self.max_new = name, batch, max_new
        self.tag = name.split("/")[-1]
        self.path = CACHE / f"{self.tag}.json"
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.model = None
        self.calls = 0
        self.seconds = 0.0

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(self.name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if torch.cuda.is_available():
            # 4-bit: this box's GPU is a shared MIG slice, so keep the footprint small and robust to contention.
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            self.model = load_with_retry(lambda: AutoModelForCausalLM.from_pretrained(
                self.name, quantization_config=qcfg, device_map={"": 0}).eval())
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=torch.float32).eval()

    def _key(self, system, user):
        return hashlib.md5((self.tag + system + "||" + user).encode("utf-8")).hexdigest()

    def gen(self, system: str, users: list[str]) -> list[str]:
        keys = [self._key(system, u) for u in users]
        miss = [i for i, k in enumerate(keys) if k not in self.cache]
        if miss:
            import torch
            self._load()
            t0 = time.perf_counter()
            for s in range(0, len(miss), self.batch):
                idx = miss[s:s + self.batch]
                prompts = []
                for i in idx:
                    msgs = [{"role": "system", "content": system}, {"role": "user", "content": users[i]}]
                    prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
                enc = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    out = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
                for j, i in enumerate(idx):
                    txt = self.tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    self.cache[keys[i]] = txt.split("\n")[0].strip().strip('"')
            self.calls += len(miss)
            self.seconds += time.perf_counter() - t0
            self.save()
        return [self.cache[k] for k in keys]

    def save(self):
        self.path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")

    def free(self):
        import torch, gc
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


def rewrite_prompt(window: list[dict], msg: str) -> str:
    lines = [f"[{h['source']}] {h['text']}" for h in window]
    return "Recent conversation events (oldest first):\n" + "\n".join(lines) + f"\n\nCustomer's latest message: {msg}\nStandalone search query:"


def resolver_prompt(tasks: list[dict], notifs: list[dict], msg: str) -> str:
    lines = []
    for t in tasks[-3:]:
        lines.append(f"- topic={t['family']}; product={t['product'] or 'unknown'}; earlier question=\"{t['last_q'] or '-'}\"")
    nl = [f"- \"{n['text'][:110]}\"" for n in notifs[-2:]]
    return ("Customer's active tasks (most recent last):\n" + ("\n".join(lines) or "- none") +
            "\nRecent service notifications:\n" + ("\n".join(nl) or "- none") +
            f"\n\nCustomer's new message: {msg}\nStandalone search query:")
