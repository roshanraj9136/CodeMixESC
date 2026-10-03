"""Experience retrieval over the ESConv case bank (get_strategy() of the base paper).

The case bank is the list of (seeker post, supporter response, strategy) triples of the
training conversations (dataset[100:]). It is encoded once per encoder and cached as .npy,
so swapping the encoder (English -> multilingual -> fine-tuned) is a one-line change.
"""
import hashlib
import os
import threading

import numpy as np
import torch

from .esconv import ROOT, case_bank

ENCODERS = {
    "roberta": "sentence-transformers/all-roberta-large-v1",  # base paper (English only)
    "labse": "sentence-transformers/LaBSE",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # ours, before fine-tuning
    "mpnet-ft": os.path.join(ROOT, "models", "codemix-retriever"),  # ours, contrastively fine-tuned
}
EMB_DIR = os.path.join(ROOT, "cache", "emb")
_MODELS = {}
_LOCK = threading.Lock()


def device():
    return os.environ.get("CODEMIX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(name):
    from sentence_transformers import SentenceTransformer
    path = ENCODERS.get(name, name)
    with _LOCK:
        if path not in _MODELS:
            _MODELS[path] = SentenceTransformer(path, device=device())
        return _MODELS[path]


class Retriever:
    def __init__(self, encoder="roberta", bank=None, exclude_conv=()):
        self.name = encoder
        self.model = load_encoder(encoder)
        self.bank = bank if bank is not None else case_bank(exclude_conv=set(exclude_conv))
        self.lock = threading.Lock()
        self.emb = self._bank_embeddings()

    def _bank_embeddings(self):
        os.makedirs(EMB_DIR, exist_ok=True)
        sig = hashlib.md5("\n".join(b["post"] for b in self.bank).encode("utf-8")).hexdigest()[:10]
        tag = self.name if self.name in ENCODERS else hashlib.md5(self.name.encode()).hexdigest()[:8]
        if self.name == "mpnet-ft":  # a re-trained checkpoint must not reuse stale embeddings
            st = os.path.getmtime(os.path.join(ENCODERS["mpnet-ft"], "config.json"))
            tag += f"-{int(st)}"
        path = os.path.join(EMB_DIR, f"{tag}-{sig}.npy")
        if os.path.exists(path):
            return np.load(path)
        emb = self.encode([b["post"] for b in self.bank], batch_size=128)
        tmp = path + f".{os.getpid()}.tmp.npy"
        np.save(tmp, emb)
        os.replace(tmp, path)
        return emb

    def encode(self, texts, batch_size=64):
        with self.lock:
            return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def topk_many(self, queries, k=10):
        q = self.encode(queries)
        sims = q @ self.emb.T
        idx = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        return idx, np.take_along_axis(sims, idx, axis=1)

    def topk(self, query, k=10):
        idx, sc = self.topk_many([query], k)
        return idx[0].tolist(), sc[0].tolist()

    def pairs(self, query, k=10):
        """(post, '[strategy] response') pairs exactly as get_strategy() builds them."""
        idx, _ = self.topk(query, k)
        return [(self.bank[i]["post"], f"[{self.bank[i]['strategy']}] {self.bank[i]['response']}") for i in idx], idx
