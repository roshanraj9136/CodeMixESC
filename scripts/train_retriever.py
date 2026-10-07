"""Fine-tune the cross-lingual retriever (SPEC: Method 2).

paraphrase-multilingual-mpnet-base-v2 is trained with the in-batch Multiple Negatives Ranking
loss on Hinglish -> English pairs (anchor = Hinglish, positive = its English original, the other
English sentences of the batch are the negatives):
    L = -1/B sum_i log( exp(cos(h_i, e_i)/tau) / sum_j exp(cos(h_i, e_j)/tau) ),  scale = 1/tau.
Only the query side changes: the English ESConv case bank is re-encoded once and never translated.

Memory. The model has ~278M parameters, ~192M of them in the XLM-R word-embedding matrix
(250k tokens x 768). Full AdamW fine-tuning keeps fp32 weights, gradients and two Adam moments,
16 bytes per parameter = ~4.4 GB before activations, so it cannot fit a 4 GB RTX 3050 Ti.
--freeze_embeddings auto freezes the word embeddings when the GPU has < 10 GB (or there is no
GPU): 86M trainable parameters, ~2.1 GB static. Freezing also keeps the rows of the many tokens
that never occur in the pairs (other languages and scripts) aligned with the rows that do.
fp16 autocast (CUDA), max_seq_length 128, --gradient_checkpointing and --cached
(CachedMultipleNegativesRankingLoss: the same loss and the same in-batch negatives, computed in
mini-batches) bound the activations.

Batching. In-batch negatives must not contain the anchor's own translation, so batches are built
without duplicate texts (BatchSamplers.NO_DUPLICATES). The NoDuplicatesBatchSampler of
sentence-transformers 3.3.1 keeps its shuffled indices in a set, which iterates small integers
in ascending order, so every epoch would see the same batches in file order; the sampler below
keeps the shuffle (re-seeded per epoch) and also compares normalised texts.

Checkpoint selection (never on test). Every ~1/4 epoch the model retrieves from the case bank
WITHOUT the dev conversations for every Hinglish post of ESConv-HiEn dev (light + heavy):
primary = mean(Overlap@10 with its own top-10 for the parallel English post, problem-type
Precision@10). Hinglish->English accuracy@1 / MRR@10 on data/pairs/val.jsonl are logged too and
become the primary metric (with a warning) when the dev files are missing.

Outputs: models/codemix-retriever (SentenceTransformer format; ENCODERS["mpnet-ft"]) and
results/retrieval/train_log.json.
Usage: python scripts/train_retriever.py            (smoke test: --base_model <tiny> --max_steps 4)
"""
import argparse
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np

# CODEMIX_DEVICE=cpu on a machine with a GPU: hide the GPU before torch loads, otherwise the HF
# Trainer still moves the model to cuda:0 while the evaluator feeds CPU tensors.
if os.environ.get("CODEMIX_DEVICE") == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" would unset the variable on Windows
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc.esconv import ROOT  # noqa: E402
from codemixesc.retrieval_eval import (available_versions, build_queries, overlap_at_k,  # noqa: E402
                                       pair_retrieval_metrics, problem_type_precision_at_k, split_bank,
                                       text_key, topk_from_embeddings)

from datasets import Dataset  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sentence_transformers.evaluation import SentenceEvaluator  # noqa: E402
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss  # noqa: E402
from sentence_transformers.sampler import SetEpochMixin  # noqa: E402
from sentence_transformers.trainer import SentenceTransformerTrainer  # noqa: E402
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments  # noqa: E402
from torch.utils.data import BatchSampler  # noqa: E402
from transformers import TrainerCallback, set_seed  # noqa: E402

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
PAIRS_DIR = os.path.join(ROOT, "data", "pairs")
OUT_DIR = os.path.join(ROOT, "models", "codemix-retriever")
CKPT_DIR = os.path.join(ROOT, "models", "codemix-retriever-ckpt")
LOG_PATH = os.path.join(ROOT, "results", "retrieval", "train_log.json")
FREEZE_BELOW_GB = 10.0
K = 10


# ============================================================================ batching
class ShuffledNoDuplicatesBatchSampler(SetEpochMixin, BatchSampler):
    """NO_DUPLICATES batches that are really shuffled: an insertion-ordered dict keeps the random
    permutation (re-seeded with seed + epoch, so a resumed run sees the same batches), and a
    pair joins a batch only if none of its normalised texts (text_key) is already in it."""

    def __init__(self, dataset, batch_size, drop_last, seed=42):
        super().__init__(dataset, batch_size, drop_last)
        cols = list(dataset.column_names)
        self.keys = [frozenset(text_key(t) for t in row) for row in zip(*(dataset[c] for c in cols))]
        self.batch_size, self.drop_last, self.seed = batch_size, drop_last, seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        remaining = dict.fromkeys(torch.randperm(len(self.keys), generator=g).tolist())
        while remaining:
            batch, seen = [], set()
            for i in remaining:
                if self.keys[i] & seen:
                    continue
                batch.append(i)
                seen |= self.keys[i]
                if len(batch) == self.batch_size:
                    break
            for i in batch:
                del remaining[i]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        n = len(self.keys)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)


class PairTrainer(SentenceTransformerTrainer):
    def get_batch_sampler(self, dataset, batch_size, drop_last, valid_label_columns=None, generator=None):
        if self.args.batch_sampler == BatchSamplers.NO_DUPLICATES and isinstance(dataset, Dataset):
            return ShuffledNoDuplicatesBatchSampler(dataset, batch_size, drop_last, seed=self.args.seed)
        return super().get_batch_sampler(dataset, batch_size, drop_last, valid_label_columns, generator)


class EvalAtEnd(TrainerCallback):
    """Evaluate (and save) at the last step too, so the final weights compete for 'best'."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


# ============================================================================ evaluation
class DevRetrievalEvaluator(SentenceEvaluator):
    """Checkpoint selection on ESConv-HiEn dev (Hinglish queries, case bank without the dev
    conversations) and on the held-out Hinglish-English pairs."""

    def __init__(self, dev_queries, bank, val_pairs, k=K, batch_size=64, name="cmx"):
        super().__init__()
        self.dev, self.bank, self.val = dev_queries, bank, val_pairs
        self.k, self.batch_size, self.name = k, batch_size, name
        self.primary_metric = "primary"
        self.greater_is_better = True
        self.levels = sorted({q["version"] for q in dev_queries})

    def _encode(self, model, texts):
        return model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)

    def compute(self, model):
        k, m = self.k, {}
        if self.dev:
            bank_emb = self._encode(model, [b["post"] for b in self.bank])
            texts = sorted({q["post"] for q in self.dev} | {q["post_en"] for q in self.dev})
            top = dict(zip(texts, topk_from_embeddings(self._encode(model, texts), bank_emb, k)))
            for level in self.levels + [None]:
                Q = [q for q in self.dev if level is None or q["version"] == level]
                R = np.stack([top[q["post"]] for q in Q])
                E = np.stack([top[q["post_en"]] for q in Q])
                pre = "dev" if level is None else f"dev_{level}"
                m[f"{pre}_overlap@{k}"] = overlap_at_k(R, E, k).mean()
                m[f"{pre}_p@{k}"] = problem_type_precision_at_k(R, [q["problem_type"] for q in Q], self.bank, k).mean()
                if level is None:  # collapse diagnostic: distinct cases among all retrieved ones
                    m[f"dev_unique@{k}"] = len(np.unique(R)) / R.size
            m["dev_primary"] = (m[f"dev_overlap@{k}"] + m[f"dev_p@{k}"]) / 2
        if self.val:
            en = [p["en"] for p in self.val]
            acc1, mrr = pair_retrieval_metrics(self._encode(model, [p["hi"] for p in self.val]),
                                               self._encode(model, en), en, k)
            m["val_acc@1"], m[f"val_mrr@{k}"] = acc1, mrr
        m["primary"] = m["dev_primary"] if self.dev else m[f"val_mrr@{k}"]
        return {key: float(v) for key, v in m.items()}

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        metrics = self.prefix_name_to_metrics(self.compute(model), self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics


# ============================================================================ helpers
def pick_device():
    return os.environ.get("CODEMIX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def gpu_memory_gb(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_properties(torch.device(device).index or 0).total_memory / 2 ** 30
    return 0.0


def load_pairs(path, limit=0, seed=42):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def dev_queries(limit=0):
    """Hinglish dev queries (light + heavy) with their parallel English posts; [] if missing."""
    out = []
    for level in available_versions("dev", ("light", "heavy")):
        q = build_queries("dev", level)
        out.extend(q[:limit] if limit else q)
    return out


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def portable(path):
    """Repo-relative path for the logs; the bare file name outside the repo (no user directories)."""
    path = os.path.abspath(path)
    try:
        rel = os.path.relpath(path, ROOT)  # case-insensitive on Windows
    except ValueError:  # another drive
        return os.path.basename(path)
    return os.path.basename(path) if rel.startswith("..") else rel.replace(os.sep, "/")


def eval_history(log_history, name="cmx"):
    pre = f"eval_{name}_"
    return [{"step": h["step"], "epoch": round(h.get("epoch", 0.0), 4),
             **{key[len(pre):]: v for key, v in h.items() if key.startswith(pre)}}
            for h in log_history if pre + "primary" in h]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--pairs_dir", default=PAIRS_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--ckpt_dir", default=CKPT_DIR)
    ap.add_argument("--log", default=LOG_PATH)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--scale", type=float, default=20.0, help="1/tau of the MNRL softmax (20 = tau 0.05)")
    ap.add_argument("--max_seq_length", type=int, default=128)
    ap.add_argument("--freeze_embeddings", choices=["auto", "yes", "no"], default="auto",
                    help=f"freeze the word embeddings (auto: when GPU memory < {FREEZE_BELOW_GB:g} GB)")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--cached", action="store_true", help="CachedMultipleNegativesRankingLoss (GradCache)")
    ap.add_argument("--mini_batch_size", type=int, default=16, help="forward mini-batch of --cached")
    ap.add_argument("--no_fp16", action="store_true", help="disable fp16 autocast on CUDA")
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument("--evals_per_epoch", type=int, default=4)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--limit", type=int, default=0, help="use N train pairs, N val pairs, N dev queries per level")
    ap.add_argument("--bank_limit", type=int, default=0, help="use the first N case-bank entries (smoke tests)")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --ckpt_dir")
    ap.add_argument("--keep_checkpoints", action="store_true", help="keep --ckpt_dir after a successful run")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = pick_device()
    on_cuda = device.startswith("cuda")
    gpu_gb = gpu_memory_gb(device)
    freeze = args.freeze_embeddings == "yes" or (args.freeze_embeddings == "auto" and gpu_gb < FREEZE_BELOW_GB)

    train = load_pairs(os.path.join(args.pairs_dir, "train.jsonl"), args.limit, args.seed)
    val_path = os.path.join(args.pairs_dir, "val.jsonl")
    val = load_pairs(val_path, args.limit, args.seed) if os.path.exists(val_path) else []
    dev = dev_queries(args.limit)
    if not dev:
        warnings.warn("ESConv-HiEn dev files missing: selecting the checkpoint by val MRR@10 instead of the dev "
                      "retrieval metrics", stacklevel=1)
        if not val:
            raise SystemExit(f"neither dev files nor {val_path}: nothing to select a checkpoint on")
    bank = split_bank("dev") if dev else []  # the case bank without the dev conversations
    if args.bank_limit:
        bank = bank[:args.bank_limit]
    print(f"[train] device {device} ({gpu_gb:.1f} GB), {len(train)} train / {len(val)} val pairs, "
          f"{len(dev)} dev queries, bank {len(bank)}", flush=True)

    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_length
    if freeze:
        for p in model[0].auto_model.get_input_embeddings().parameters():
            p.requires_grad_(False)
    n_total = sum(p.numel() for p in model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[train] {n_total / 1e6:.1f}M parameters, {n_frozen / 1e6:.1f}M frozen "
          f"(word embeddings {'frozen' if freeze else 'trained'})", flush=True)

    loss = (CachedMultipleNegativesRankingLoss(model, scale=args.scale, mini_batch_size=args.mini_batch_size)
            if args.cached else MultipleNegativesRankingLoss(model, scale=args.scale))
    loss_name = type(loss).__name__
    evaluator = DevRetrievalEvaluator(dev, bank, val, k=K, batch_size=args.eval_batch_size)
    train_ds = Dataset.from_dict({"anchor": [p["hi"] for p in train], "positive": [p["en"] for p in train]})

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    eval_steps = max(1, min(total_steps, round(steps_per_epoch / max(1, args.evals_per_epoch))))
    fp16 = on_cuda and not args.no_fp16
    targs = SentenceTransformerTrainingArguments(
        output_dir=args.ckpt_dir, num_train_epochs=args.epochs, max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, optim=args.optim, fp16=fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps,
        save_total_limit=args.save_total_limit, load_best_model_at_end=True,
        metric_for_best_model=f"eval_{evaluator.name}_primary", greater_is_better=True,
        logging_steps=max(1, eval_steps // 4), seed=args.seed, data_seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        use_cpu=not on_cuda, dataloader_num_workers=0, dataloader_pin_memory=on_cuda,
        report_to="none", disable_tqdm=not sys.stderr.isatty())
    trainer = PairTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=evaluator,
                          callbacks=[EvalAtEnd()])

    before = evaluator.compute(model)
    print(f"[train] before fine-tuning: {json.dumps({k: round(v, 4) for k, v in before.items()})}", flush=True)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    resume = args.resume and os.path.isdir(args.ckpt_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(args.ckpt_dir))
    out = trainer.train(resume_from_checkpoint=True if resume else None)
    train_time = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30 if on_cuda else None

    best_ckpt = trainer.state.best_model_checkpoint
    best_step = int(best_ckpt.rsplit("-", 1)[-1]) if best_ckpt else None
    best_metric = trainer.state.best_metric
    history = eval_history(trainer.state.log_history, evaluator.name)
    os.makedirs(args.out_dir, exist_ok=True)
    trainer.model.save(args.out_dir)
    del trainer, model, loss
    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()
    final = evaluator.compute(SentenceTransformer(args.out_dir, device=device))  # exactly what the pipeline loads
    print(f"[train] best step {best_step}; saved model: {json.dumps({k: round(v, 4) for k, v in final.items()})}",
          flush=True)

    log = {
        "config": {**vars(args), "pairs_dir": portable(args.pairs_dir),
                   "base_model": portable(args.base_model) if os.path.exists(args.base_model) else args.base_model,
                   "out_dir": portable(args.out_dir), "ckpt_dir": portable(args.ckpt_dir), "log": portable(args.log),
                   "tau": 1.0 / args.scale, "k": K, "eval_steps": eval_steps, "steps_per_epoch": steps_per_epoch,
                   "total_steps": total_steps, "fp16": fp16, "batch_sampler": "no_duplicates (shuffled, see docstring)",
                   "loss": loss_name},
        "data": {"train_pairs": len(train), "val_pairs": len(val), "dev_queries": len(dev), "bank": len(bank),
                 "train_by_source": {s: sum(p["src"] == s for p in train) for s in ("phinc", "esconv")},
                 "dev_levels": evaluator.levels, "selection_metric": "dev" if dev else "val_mrr@10"},
        "device": {"device": device, "gpu": torch.cuda.get_device_name(0) if on_cuda else None,
                   "gpu_memory_gb": round(gpu_gb, 2), "peak_memory_allocated_gb": round(peak_gb, 3) if peak_gb else None},
        "params": {"total": n_total, "frozen": n_frozen, "trainable": n_total - n_frozen,
                   "word_embeddings_frozen": freeze},
        "eval_before_training": before,
        "evals": history,
        "best_step": best_step,
        "best_metric": best_metric,
        "final_eval": final,
        "train_time_s": round(train_time, 1),
        "train_metrics": out.metrics,
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                     "sentence_transformers": __import__("sentence_transformers").__version__},
        "git_commit": git_commit(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "w", encoding="utf-8", newline="\n") as f:
        json.dump(log, f, indent=1)
    with open(os.path.join(args.out_dir, "codemix_training.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({key: log[key] for key in ("config", "data", "params", "best_step", "final_eval", "git_commit")},
                  f, indent=1)
    if not args.keep_checkpoints:
        try:
            shutil.rmtree(args.ckpt_dir)
        except OSError as e:  # e.g. a file still memory-mapped on Windows
            print(f"[train] could not remove {args.ckpt_dir} ({e}); delete it by hand", flush=True)
    print(f"[train] done in {train_time / 60:.1f} min; log {portable(args.log)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
