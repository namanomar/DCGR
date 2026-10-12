"""Embedding back-ends (all local). Disk-cached so every experiment re-uses vectors.

Back-ends
  tfidf        - sparse lexical vectors (sklearn), fit on KB text            (no neural net)
  minilm       - sentence-transformers/all-MiniLM-L6-v2     (22M params, 384d)
  bge-small    - BAAI/bge-small-en-v1.5                      (33M params, 384d)
  qwen3-emb    - Qwen/Qwen3-Embedding-0.6B                   (600M params, 1024d)
  e5-large     - intfloat/e5-large-v2                        (335M params, 1024d)
  gte-large    - thenlper/gte-large                          (335M params, 1024d)
  qwen3-emb-4b - Qwen/Qwen3-Embedding-4B                      (4B params, 2560d)
"""
from __future__ import annotations
import hashlib, pickle, os
from pathlib import Path
import numpy as np
from common import load_with_retry

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "emb_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "qwen3-emb": "Qwen/Qwen3-Embedding-0.6B",
    "e5-large": "intfloat/e5-large-v2",
    "gte-large": "thenlper/gte-large",
    "qwen3-emb-4b": "Qwen/Qwen3-Embedding-4B",
}
QUERY_PREFIX = {"bge-small": "Represent this sentence for searching relevant passages: ", "e5-large": "query: "}
DOC_PREFIX = {"e5-large": "passage: "}
QWEN_TASK = "Given a customer message from an insurance and mutual fund chat, retrieve the knowledge-base article that answers it"


def _h(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


class Embedder:
    def __init__(self, name: str, device: str | None = None):
        self.name = name
        self.model = None
        self.device = device
        self.cache_path = CACHE_DIR / f"{name}.pkl"
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            self.cache = pickle.loads(self.cache_path.read_bytes())
        self._dirty = False

    # --------------------------------------------------------------------
    def _load(self):
        if self.model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {}
        if self.name.startswith("qwen3-emb") and dev == "cuda":
            kw = dict(model_kwargs={"torch_dtype": torch.float16})
        self.model = load_with_retry(lambda: SentenceTransformer(HF[self.name], device=dev, **kw))
        self.model.max_seq_length = 256

    def _encode_raw(self, texts: list[str], is_query: bool) -> np.ndarray:
        self._load()
        if self.name.startswith("qwen3-emb") and is_query:
            texts = [f"Instruct: {QWEN_TASK}\nQuery: {t}" for t in texts]
        elif is_query and self.name in QUERY_PREFIX:
            texts = [QUERY_PREFIX[self.name] + t for t in texts]
        elif not is_query and self.name in DOC_PREFIX:
            texts = [DOC_PREFIX[self.name] + t for t in texts]
        v = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return v.astype(np.float32)

    def encode(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        """Cached encode. `is_query=False` for documents (no instruction prefix)."""
        keys = [("q:" if is_query else "d:") + _h(t) for t in texts]
        miss = [i for i, k in enumerate(keys) if k not in self.cache]
        if miss:
            vecs = self._encode_raw([texts[i] for i in miss], is_query)
            self._dirty = True
            for i, v in zip(miss, vecs):
                self.cache[keys[i]] = v
            self._dirty = True
        return np.stack([self.cache[k] for k in keys])

    def save(self):
        if self._dirty:
            self.cache_path.write_bytes(pickle.dumps(self.cache))
            self._dirty = False

    def free(self):
        import torch, gc
        self.save()
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TfidfEmbedder:
    """Sparse TF-IDF 'embedding' with the same interface (dense numpy, small vocab -> fine)."""
    name = "tfidf"

    def __init__(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self.vec.fit(corpus)

    def encode(self, texts, is_query=True):
        m = self.vec.transform(texts)
        return np.asarray(m.todense(), dtype=np.float32)

    def save(self): ...
    def free(self): ...


_REG: dict[str, object] = {}


def get_embedder(name: str, corpus: list[str] | None = None):
    if name == "tfidf":
        if "tfidf" not in _REG:
            _REG["tfidf"] = TfidfEmbedder(corpus)
        return _REG["tfidf"]
    if name not in _REG:
        _REG[name] = Embedder(name)
    return _REG[name]
