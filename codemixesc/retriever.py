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


def model_signature(path):
    """Fingerprint of a local model directory (weight/config file names, sizes, mtimes), so that
    a model re-trained or re-saved at the same path never reuses stale embeddings; '' for hub ids."""
    if not os.path.isdir(path):
        return ""
    files = sorted(f for f in os.listdir(path) if f.endswith((".safetensors", ".bin", "config.json")))
    info = [(f, os.path.getsize(os.path.join(path, f)), int(os.path.getmtime(os.path.join(path, f)))) for f in files]
    return hashlib.md5(repr(info).encode()).hexdigest()[:8]


def load_encoder(name):
    """(model, lock) shared by every Retriever of the same model (encode() is not re-entrant)."""
    from sentence_transformers import SentenceTransformer
    path = ENCODERS.get(name, name)
    key = (path, model_signature(path))
    with _LOCK:
        if key not in _MODELS:
            _MODELS[key] = (SentenceTransformer(path, device=device()), threading.Lock())
        return _MODELS[key]


class Retriever:
    def __init__(self, encoder="roberta", bank=None, exclude_conv=()):
        self.name = encoder
        self.model, self.lock = load_encoder(encoder)
        exclude = set(exclude_conv)
        self.bank = [b for b in bank if b.get("conv") not in exclude] if bank is not None else case_bank(exclude_conv=exclude)
        self.emb = self._bank_embeddings()

    def _bank_embeddings(self):
        os.makedirs(EMB_DIR, exist_ok=True)
        # the file name identifies both the bank (its posts) and the model (name + weight files),
        # so e.g. the dev bank (dev conversations excluded) never reuses the test bank's file
        bank_sig = hashlib.md5("\n".join(b["post"] for b in self.bank).encode("utf-8")).hexdigest()[:10]
        tag = self.name if self.name in ENCODERS else hashlib.md5(self.name.encode()).hexdigest()[:8]
        model_sig = model_signature(ENCODERS.get(self.name, self.name))
        if model_sig:  # a re-trained or re-saved local model must not reuse stale embeddings
            tag += f"-{model_sig}"
        path = os.path.join(EMB_DIR, f"{tag}-{bank_sig}.npy")
        if os.path.exists(path):
            emb = np.load(path)
            if emb.shape[0] == len(self.bank):
                return emb
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
        """(post, '[strategy] response') pairs exactly as get_strategy() builds them: the base code
        reads the case bank from a text file written with .replace("\n", "\\n"), so line breaks
        inside a post or response appear as a literal backslash-n in the prompt."""
        idx, _ = self.topk(query, k)
        esc = lambda t: t.replace("\n", "\\n")  # noqa: E731
        return [(esc(self.bank[i]["post"]), f"[{self.bank[i]['strategy']}] {esc(self.bank[i]['response'])}")
                for i in idx], idx
