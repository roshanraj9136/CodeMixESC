import os
import re
from collections import Counter

import numpy as np

from .esconv import (STRATEGIES, case_bank, dev_ids, hien_dir, load_dev, load_esconv, load_version,
                     turn_samples)

K = 10
_WORD_RE = re.compile(r"\w+")


def text_key(text):
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _rows(x):
    a = np.asarray(x)
    return (a[None, :], True) if a.ndim == 1 else (a, False)


def topk_from_embeddings(q_emb, bank_emb, k=K):
    sims = np.asarray(q_emb, dtype=np.float32) @ np.asarray(bank_emb, dtype=np.float32).T
    return np.argsort(-sims, axis=1, kind="stable")[:, :k]


def overlap_at_k(a, b, k=K):
    A, single = _rows(a)
    B, _ = _rows(b)
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"{A.shape[0]} vs {B.shape[0]} queries")
    if A.shape[1] < k or B.shape[1] < k:
        raise ValueError(f"need at least k={k} retrieved cases per query")
    out = np.array([len(set(x[:k].tolist()) & set(y[:k].tolist())) / k for x, y in zip(A, B)])
    return float(out[0]) if single else out


def problem_type_precision_at_k(retrieved_idx, query_problem_types, bank, k=K):
    R, single = _rows(retrieved_idx)
    if R.shape[1] < k:
        raise ValueError(f"need at least k={k} retrieved cases per query")
    pts = [query_problem_types] if isinstance(query_problem_types, str) else list(query_problem_types)
    if len(pts) != R.shape[0]:
        raise ValueError(f"{len(pts)} problem types for {R.shape[0]} queries")
    bank_pt = np.array([b["problem_type"] for b in bank], dtype=object)
    hits = bank_pt[R[:, :k]] == np.array(pts, dtype=object)[:, None]
    out = hits.mean(axis=1).astype(float)
    return float(out[0]) if single else out


def chance_precision(query_problem_types, bank):
    counts = Counter(b["problem_type"] for b in bank)
    return np.array([counts.get(p, 0) / len(bank) for p in query_problem_types], dtype=float)


def strategy_labels(bank):
    return list(STRATEGIES) + sorted({b["strategy"] for b in bank} - set(STRATEGIES))


def strategy_histograms(retrieved_idx, bank, k=K, labels=None):
    labels = labels or strategy_labels(bank)
    pos = {s: i for i, s in enumerate(labels)}
    bank_lab = np.array([pos[b["strategy"]] for b in bank])
    R, single = _rows(retrieved_idx)
    H = np.zeros((R.shape[0], len(labels)))
    np.add.at(H, (np.arange(R.shape[0])[:, None], bank_lab[R[:, :k]]), 1.0)
    return H[0] if single else H


def jsd(p, q, base=2):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("empty histogram")
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log(a[nz] / b[nz])))

    return float(min(1.0, max(0.0, (kl(p, m) + kl(q, m)) / 2 / np.log(base))))


def strategy_jsd(idx_a, idx_b, bank, k=K):
    labels = strategy_labels(bank)
    Ha, Hb = strategy_histograms(idx_a, bank, k, labels), strategy_histograms(idx_b, bank, k, labels)
    Ha, Hb = np.atleast_2d(Ha), np.atleast_2d(Hb)
    return jsd(Ha.sum(0), Hb.sum(0)), np.array([jsd(a, b) for a, b in zip(Ha, Hb)])


def pair_retrieval_metrics(hi_emb, en_emb, en_texts, k=K):
    keys, first, gold = {}, [], []
    for i, t in enumerate(en_texts):
        kk = text_key(t)
        if kk not in keys:
            keys[kk] = len(first)
            first.append(i)
        gold.append(keys[kk])
    sims = np.asarray(hi_emb, dtype=np.float32) @ np.asarray(en_emb, dtype=np.float32)[first].T
    gold = np.array(gold)
    gold_score = sims[np.arange(len(gold)), gold]
    rank = (sims >= gold_score[:, None]).sum(axis=1)
    return float(np.mean(rank == 1)), float(np.mean(np.where(rank <= k, 1.0 / rank, 0.0)))


def bootstrap_ci(values, groups=None, n_boot=2000, alpha=0.05, seed=42, stat=None):
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    n = len(v)
    if n == 0:
        return {"value": None, "lo": None, "hi": None, "n": 0, "n_groups": 0}
    g = np.arange(n) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1].ravel()
    G = int(g.max()) + 1
    S = np.zeros((G, v.shape[1]))
    np.add.at(S, g, v)
    C = np.bincount(g, minlength=G).astype(float)
    W = np.random.default_rng(seed).multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)
    SB, CB = W @ S, W @ C
    if stat is None:
        if v.shape[1] != 1:
            raise ValueError("give a stat for vector-valued data")
        value, boot = S.sum() / C.sum(), SB[:, 0] / CB
    else:
        value, boot = stat(S.sum(0), C.sum()), np.array([stat(s, c) for s, c in zip(SB, CB)])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"value": float(value), "lo": float(lo), "hi": float(hi), "n": int(n), "n_groups": G}


def split_conversations(split, version):
    if split == "test":
        return list(enumerate(load_version(version)))
    if split == "dev":
        return [(c["esconv_index"], c) for c in load_dev(version)]
    raise ValueError(f"unknown split {split!r}")


def available_versions(split, versions=("en", "light", "heavy")):
    out = []
    for v in versions:
        if split == "test" and v == "en":
            out.append(v)
            continue
        name = "dev_conv_ids.json" if v == "en" else f"{split}_{v}.json"
        if os.path.exists(os.path.join(hien_dir(), name)):
            out.append(v)
    return out


def split_bank(split):
    return case_bank(exclude_conv=set(dev_ids()) if split == "dev" else ())


def build_queries(split="test", version="en"):
    data, out = None, []
    for cid, conv in split_conversations(split, version):
        dialog = conv["dialog"]
        for s in turn_samples(conv, cid):
            prev = dialog[s["turn"] - 1]
            if prev["content"].strip() != s["post"]:
                raise AssertionError(f"{split}/{version} conv {cid}: post is not dialog turn {s['turn'] - 1}")
            if version == "en":
                post_en = s["post"]
            elif "content_en" in prev:
                post_en = prev["content_en"].strip()
            else:
                raise KeyError(f"{split}/{version} conv {cid} turn {s['turn'] - 1} has no content_en")
            pt = s["problem_type"]
            if pt is None:
                data = data or load_esconv()
                pt = data[cid]["problem_type"]
            out.append({"uid": s["uid"], "conv_id": cid, "turn": s["turn"], "post": s["post"],
                        "post_en": post_en, "problem_type": pt, "early": s["early"], "version": version})
    return out
