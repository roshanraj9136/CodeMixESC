"""Retrieval robustness of the experience retriever (SPEC: Metrics, "Retrieval robustness").

Queries are the `post` of every supporter turn (exactly what get_strategy() sends to the
retriever): all 1,210 test turns and the fixed sampled_uids(200) subset the multi-agent systems
run on, in every available version (en, light, heavy). For each encoder and version:
  P@10                   problem-type Precision@10 of the retrieved cases
  Overlap@10 (self)      top-10 of the Hinglish post vs top-10 of its parallel English post,
                         same encoder (light/heavy only)
  Overlap@10 vs RoBERTa  top-10 of the post vs the base paper's retriever (all-roberta-large-v1)
                         on the parallel English post: does the system still see the experience
                         MultiAgentESC sees in English?
  Strategy JSD           JSD (base 2) between the strategy histograms of the cases retrieved for
                         the Hinglish and the English posts, pooled over the queries (a systematic
                         shift of the strategy evidence; per-query JSD is in the JSON)
95% CIs: cluster bootstrap over conversations; differences to RoBERTa are paired (same
resamples). --split dev evaluates ESConv-HiEn dev with the dev conversations removed from the bank.

Outputs: results/retrieval/retrieval_{split}.json, results/tables/retrieval_{split}.md and the
booktabs tabulars (no float, ready for \\input): retrieval_{split}.tex (P@10 and Overlap@10),
retrieval_{split}_full.tex (+ strategy JSD) and, for test, retrieval_test_s200.tex.
Usage: python scripts/eval_retrieval.py --split test [--encoders roberta,mpnet,mpnet-ft]
       python scripts/eval_retrieval.py --stub      (hashing encoder, no downloads; tests)
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc import retriever as retriever_mod  # noqa: E402
from codemixesc.esconv import ROOT, sampled_uids  # noqa: E402
from codemixesc.retrieval_eval import (K, available_versions, bootstrap_ci, build_queries,  # noqa: E402
                                       chance_precision, jsd, overlap_at_k, problem_type_precision_at_k,
                                       split_bank, strategy_histograms, strategy_labels)
from codemixesc.retriever import ENCODERS, Retriever  # noqa: E402

ALL_ENCODERS = ["roberta", "labse", "mpnet", "mpnet-ft"]  # roberta first: the reference for the others
DISPLAY = {"roberta": "RoBERTa-L (base)", "labse": "LaBSE", "mpnet": "mMPNet", "mpnet-ft": "mMPNet-FT (ours)"}
LEGEND = ("RoBERTa-L = all-roberta-large-v1, the English retriever of MultiAgentESC; mMPNet = "
          "paraphrase-multilingual-mpnet-base-v2, FT = our contrastive (MNRL) fine-tuning on Hinglish-English pairs")
VERSIONS = ["en", "light", "heavy"]
VNAME = {"en": "EN", "light": "Light", "heavy": "Heavy"}
STUB_DIMS = {"roberta": 64, "labse": 96, "mpnet": 128, "mpnet-ft": 256}
# (metric, version, scale, decimals, higher is better, group, subgroup)
P10 = [("p@10", v, 100, 1, True, "P@10", "") for v in VERSIONS]
SELF = [("overlap_self@10", v, 100, 1, True, "Overlap@10", "self") for v in ("light", "heavy")]
VSROB = [("overlap_roberta_en@10", v, 100, 1, True, "Overlap@10", "vs. RoBERTa-EN") for v in VERSIONS]
JSD = [("jsd_strategy", v, 100, 2, False, "Strategy JSD", "") for v in ("light", "heavy")]
MAIN_COLUMNS, FULL_COLUMNS = P10 + SELF + VSROB, P10 + SELF + VSROB + JSD


def make_retriever(name, bank, stub=False):
    """A Retriever over `bank` (the full case bank, or the full bank minus some conversations, in
    case-bank order). The bank embeddings always come from the FULL-bank cache file, the one the
    test-time pipeline uses, and are subset by row: the dev evaluation never writes a second cache
    file, and a cached matrix that does not match the bank is recomputed in memory."""
    if stub:  # the real Retriever ranking code over a hashing encoder; nothing is cached on disk
        from codemixesc.testing import HashEncoder
        r = Retriever.__new__(Retriever)
        r.name, r.model, r.lock, r.bank = f"stub-{name}", HashEncoder(STUB_DIMS.get(name, 200)), threading.Lock(), bank
        r.emb = r.encode([b["post"] for b in bank], batch_size=128)
        return r
    if name == "mpnet-ft" and not os.path.isdir(ENCODERS["mpnet-ft"]):
        raise FileNotFoundError(f"{ENCODERS['mpnet-ft']} does not exist; run scripts/train_retriever.py first")
    r = Retriever(name)
    if r.emb.shape[0] != len(r.bank):
        warnings.warn(f"{name}: cached bank embeddings have {r.emb.shape[0]} rows for a bank of {len(r.bank)}; "
                      "recomputing them in memory", stacklevel=1)
        r.emb = r.encode([b["post"] for b in r.bank], batch_size=128)
    if len(bank) != len(r.bank):
        keep_conv = {b["conv"] for b in bank}
        keep = [i for i, b in enumerate(r.bank) if b["conv"] in keep_conv]
        r.bank, r.emb = [r.bank[i] for i in keep], r.emb[keep]
    if r.bank != bank:
        raise AssertionError(f"{name}: retriever bank differs from the evaluation bank")
    return r


def free_models():
    retriever_mod._MODELS.clear()  # one encoder at a time on a 4 GB GPU
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def topk_texts(r, texts, k, chunk=512):
    out = [r.topk_many(texts[s:s + chunk], k)[0] for s in range(0, len(texts), chunk)]
    return dict(zip(texts, np.concatenate(out))) if out else {}


def evaluate(tops, queries, subsets, bank, k, n_boot, seed):
    """results[encoder][version][subset][metric], chance[version][subset], deltas vs roberta."""
    labels = strategy_labels(bank)
    L = len(labels)

    def ci(vals, groups, stat=None):
        return bootstrap_ci(vals, groups, n_boot=n_boot, seed=seed, stat=stat)

    def pooled(s, c):
        return jsd(s[:L], s[L:])

    def pooled_diff(s, c):
        return jsd(s[:L], s[L:2 * L]) - jsd(s[2 * L:3 * L], s[3 * L:])

    results, chance, deltas, per_query = {}, {}, {}, {}
    for v, qs in queries.items():
        for sname, uids in subsets.items():
            Q = [q for q in qs if uids is None or q["uid"] in uids]
            if not Q:
                continue
            groups = [q["conv_id"] for q in Q]
            pts = [q["problem_type"] for q in Q]
            chance.setdefault(v, {})[sname] = ci(chance_precision(pts, bank), groups)
            for enc, top in tops.items():
                Rv = np.stack([top[q["post"]] for q in Q])
                Ren = np.stack([top[q["post_en"]] for q in Q])
                pq = {"p@10": problem_type_precision_at_k(Rv, pts, bank, k)}
                if v != "en":
                    pq["overlap_self@10"] = overlap_at_k(Rv, Ren, k)
                    pq["hist"] = np.hstack([strategy_histograms(Rv, bank, k, labels),
                                            strategy_histograms(Ren, bank, k, labels)])
                    pq["jsd_strategy_per_query"] = np.array([jsd(h[:L], h[L:]) for h in pq["hist"]])
                if "roberta" in tops:
                    pq["overlap_roberta_en@10"] = overlap_at_k(Rv, np.stack([tops["roberta"][q["post_en"]] for q in Q]), k)
                per_query[(enc, v, sname)] = pq
                res = {m: ci(x, groups) for m, x in pq.items() if m != "hist"}
                if "hist" in pq:
                    res["jsd_strategy"] = ci(pq["hist"], groups, pooled)
                res["n"] = len(Q)
                results.setdefault(enc, {}).setdefault(v, {})[sname] = res
            if "roberta" not in tops:
                continue
            ref = per_query[("roberta", v, sname)]
            for enc in tops:
                if enc == "roberta":
                    continue
                pq = per_query[(enc, v, sname)]
                d = {m: ci(pq[m] - ref[m], groups) for m in ("p@10", "overlap_self@10", "overlap_roberta_en@10")
                     if m in pq}
                if "hist" in pq:
                    d["jsd_strategy"] = ci(np.hstack([pq["hist"], ref["hist"]]), groups, pooled_diff)
                deltas.setdefault(enc, {}).setdefault(v, {})[sname] = d
    return results, chance, deltas


# ============================================================================ tables
def _get(part, enc, version, sname, metric):
    return part.get(enc, {}).get(version, {}).get(sname, {}).get(metric)


def _significant(report, enc, metric, version, sname):
    """The paired-bootstrap 95% CI of the difference to RoBERTa excludes 0."""
    d = _get(report["delta_vs_roberta"], enc, version, sname, metric)
    return bool(d and d.get("value") is not None and (d["lo"] > 0 or d["hi"] < 0))


def _cell(m, scale, dec, tex=False, bold=False, mark=False):
    """Markdown: 'mean ± half-width'; LaTeX: the mean only (the caption states the largest half-width)."""
    if not m or m.get("value") is None:
        return "--"
    v, hw = m["value"] * scale, (m["hi"] - m["lo"]) / 2 * scale
    val = f"{v:.{dec}f}"
    if tex:
        return (f"\\textbf{{{val}}}" if bold else val) + ("$^\\dagger$" if mark else "")
    val = f"**{val}**" if bold else val
    return (val if hw == 0 else f"{val} ± {hw:.{dec}f}") + (" †" if mark else "")


def _best(results, encoders, col, sname):
    """Encoders tied (at display precision) with the best one; the trivial RoBERTa-vs-itself cell
    does not compete."""
    metric, version, scale, dec, higher = col[:5]
    vals = {}
    for enc in encoders:
        m = _get(results, enc, version, sname, metric)
        trivial = enc == "roberta" and metric == "overlap_roberta_en@10" and version == "en"
        if m and m.get("value") is not None and not trivial:
            vals[enc] = round(m["value"] * scale, dec)
    if len(vals) < 2:
        return set()
    top = (max if higher else min)(vals.values())
    return {e for e, x in vals.items() if x == top}


def _rows(report, encoders, sname, columns, tex):
    results, rows = report["results"], []
    best = [_best(results, encoders, col, sname) for col in columns]
    for enc in encoders:
        rows.append([DISPLAY.get(enc, enc)] + [
            _cell(_get(results, enc, col[1], sname, col[0]), col[2], col[3], tex, enc in b,
                  _significant(report, enc, col[0], col[1], sname)) for col, b in zip(columns, best)])
    rows.append(["Random (chance)"] + [_cell(report["chance"].get(col[1], {}).get(sname), col[2], col[3], tex)
                                       if col[0] == "p@10" else "--" for col in columns])
    return rows


def _runs(labels):
    """Run-length groups of consecutive equal labels: [(label, span), ...]."""
    out = []
    for lab in labels:
        if out and out[-1][0] == lab:
            out[-1][1] += 1
        else:
            out.append([lab, 1])
    return out


def _max_half_width(report, encoders, sname, columns, jsd_metric):
    hws = [(m["hi"] - m["lo"]) / 2 * col[2] for col in columns if (col[0] == "jsd_strategy") == jsd_metric
           for enc in encoders if (m := _get(report["results"], enc, col[1], sname, col[0])) and m.get("value") is not None]
    return max(hws) if hws else None


def markdown(report, encoders):
    split, k = report["split"], report["k"]
    lines = [f"# Retrieval robustness ({split})", "",
             f"k = {k}; case bank of {report['bank_size']} posts"
             + (" (dev conversations excluded)" if split == "dev" else "") + "; mean ± half-width of the 95% "
             "cluster-bootstrap CI (conversations resampled). P@10 / Overlap in %, strategy JSD (base 2, pooled "
             "over the queries) ×10⁻². Overlap@10 self: Hinglish post vs its English original, same encoder. "
             "Overlap@10 vs. RoBERTa-EN: the post vs the base paper's retriever on the English original. Bold = best "
             "encoder per column; † = differs from RoBERTa-L (paired bootstrap CI of the difference excludes 0). "
             + LEGEND + ".", ""]
    if report["skipped_encoders"]:
        lines += ["Skipped encoders: " + "; ".join(f"{e} ({why})" for e, why in report["skipped_encoders"].items()), ""]
    head = ["Encoder"] + [" ".join(x for x in (c[5], c[6], VNAME[c[1]]) if x) for c in FULL_COLUMNS]
    for sname, n in report["subsets"].items():
        title = {"all": f"All {split} turns", "s200": "sampled_uids(200)"}.get(sname, sname)
        lines += [f"## {title} ({', '.join(f'{VNAME[v]} n={c}' for v, c in n.items())})", "",
                  "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(r) + " |" for r in _rows(report, encoders, sname, FULL_COLUMNS, False)]
        lines.append("")
        if report["delta_vs_roberta"]:
            lines += ["Difference to RoBERTa-L, the base paper's retriever (paired; [95% CI]):", "",
                      "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            for enc in encoders:
                if enc not in report["delta_vs_roberta"]:
                    continue
                cells = []
                for metric, v, sc, dec, *_ in FULL_COLUMNS:
                    d = _get(report["delta_vs_roberta"], enc, v, sname, metric)
                    cells.append("--" if not d or d.get("value") is None else
                                 f"{d['value'] * sc:+.{dec}f} [{d['lo'] * sc:+.{dec}f}, {d['hi'] * sc:+.{dec}f}]")
                lines.append("| " + " | ".join([DISPLAY.get(enc, enc)] + cells) + " |")
            lines.append("")
    return "\n".join(lines)


def latex(report, encoders, sname, columns):
    """A booktabs tabular (no float: the paper supplies table/caption/label) with a suggested
    caption in a leading comment."""
    split, n = report["split"], report["subsets"][sname]
    turns = max(n.values()) if n else 0
    arrow = {"Strategy JSD": " $\\downarrow$"}
    header = []
    for level in (5, 6):
        cells, rules, col = [], [], 2
        for label, c in _runs([cl[level] for cl in columns]):
            text = label.replace("vs. ", "vs.\\ ") + (arrow.get(label, " $\\uparrow$") if level == 5 else "")
            cells.append(f"\\multicolumn{{{c}}}{{c}}{{{text}}}" if label else " & ".join([""] * c))
            if label:
                rules.append(f"\\cmidrule(lr){{{col}-{col + c - 1}}}")
            col += c
        if level == 5 or rules:
            header += [" & " + " & ".join(cells) + " \\\\", "".join(rules)]
    header.append("Encoder & " + " & ".join(VNAME[c[1]] for c in columns) + " \\\\")
    body = [" & ".join(r) + " \\\\" for r in _rows(report, encoders, sname, columns, True)]
    body.insert(len(body) - 1, "\\midrule")
    what = {"all": f"all {split} turns", "s200": "the 200-turn subset of the multi-agent systems"}.get(sname, sname)
    hw_pct = _max_half_width(report, encoders, sname, columns, False)
    hw_jsd = _max_half_width(report, encoders, sname, columns, True)
    caption = (f"Retrieval robustness on ESConv-HiEn ({what}, {turns} queries per version, k={report['k']}). "
               "P@10: problem-type precision of the retrieved cases (%). Overlap@10 self: share of the top-10 cases "
               "of the English original that the same encoder also retrieves for the Hinglish post; vs. RoBERTa-EN: "
               "share of the cases the base paper's retriever finds for the English original."
               + (" Strategy JSD (x10^-2): Jensen-Shannon divergence (base 2) between the strategy distributions of "
                  "the cases retrieved for the Hinglish and the English posts." if hw_jsd is not None else "")
               + (f" 95% cluster-bootstrap half-widths <= {hw_pct:.1f} points" if hw_pct is not None else "")
               + (f" ({hw_jsd:.2f} for JSD)" if hw_jsd is not None else "") + ". Bold: best encoder; dagger: "
               "differs from RoBERTa-L (paired bootstrap, 95%). " + LEGEND + ".")
    return "\n".join([
        f"% generated by scripts/eval_retrieval.py ({report['created']}); needs \\usepackage{{booktabs}}",
        f"% suggested caption: {caption}",
        "\\begin{tabular}{l" + "c" * len(columns) + "}", "\\toprule", *header, "\\midrule", *body,
        "\\bottomrule", "\\end{tabular}", ""])


# ============================================================================ main
def model_id(name):
    """Hub id, or the repo-relative path of a local model (no local user directories in results)."""
    path = ENCODERS.get(name, name)
    return os.path.relpath(path, ROOT).replace(os.sep, "/") if os.path.isabs(path) else path


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--encoders", default=",".join(ALL_ENCODERS))
    ap.add_argument("--versions", default=",".join(VERSIONS))
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stub", action="store_true", help="hashing encoders instead of the real models (tests)")
    ap.add_argument("--results_dir", default=None, help="default results/ (scratch/results_stub with --stub)")
    args = ap.parse_args(argv)
    results_dir = args.results_dir or (os.path.join(ROOT, "scratch", "results_stub") if args.stub
                                       else os.path.join(ROOT, "results"))

    requested = [v for v in args.versions.split(",") if v]
    versions = available_versions(args.split, requested)
    missing = [v for v in requested if v not in versions]
    if missing:
        warnings.warn(f"skipping versions without files: {missing}", stacklevel=1)
    if not versions:
        raise SystemExit("no version to evaluate")
    queries = {v: build_queries(args.split, v) for v in versions}
    bank = split_bank(args.split)
    subsets = {"all": None}
    if args.split == "test":
        subsets["s200"] = set(sampled_uids(200, seed=42))
    texts = sorted({q[f] for qs in queries.values() for q in qs for f in ("post", "post_en")})
    print(f"[retrieval] {args.split}: versions {versions}, {len(texts)} distinct posts, bank {len(bank)}", flush=True)

    names = [e for e in args.encoders.split(",") if e]
    encoders = [e for e in ALL_ENCODERS if e in names] + [e for e in names if e not in ALL_ENCODERS]
    tops, skipped = {}, {}
    for enc in encoders:
        t0 = time.time()
        try:
            tops[enc] = topk_texts(make_retriever(enc, bank, args.stub), texts, args.k)
            print(f"[retrieval] {enc}: retrieved in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001 - a missing/undownloadable model is skipped, not fatal
            warnings.warn(f"skipping encoder {enc}: {e.__class__.__name__}: {str(e)[:300]}", stacklevel=1)
            skipped[enc] = f"{e.__class__.__name__}: {str(e)[:200]}"
        finally:
            free_models()
    if not tops:
        raise SystemExit("no encoder could be loaded")
    if "roberta" not in tops:
        warnings.warn("roberta unavailable: no 'Overlap@10 vs RoBERTa-EN' and no differences to the base paper",
                      stacklevel=1)

    results, chance, deltas = evaluate(tops, queries, subsets, bank, args.k, args.n_boot, args.seed)
    used = [e for e in encoders if e in tops]
    report = {
        "split": args.split, "k": args.k, "n_boot": args.n_boot, "alpha": 0.05, "seed": args.seed, "stub": args.stub,
        "bank_size": len(bank), "versions": versions, "missing_versions": missing,
        "encoders": {e: {"display": DISPLAY.get(e, e), "model": "stub" if args.stub else model_id(e)} for e in used},
        "skipped_encoders": skipped,
        "subsets": {s: {v: sum(1 for q in queries[v] if u is None or q["uid"] in u) for v in versions}
                    for s, u in subsets.items()},
        "results": results, "chance": chance, "delta_vs_roberta": deltas,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
    }
    out_json = os.path.join(results_dir, "retrieval", f"retrieval_{args.split}.json")
    write_text(out_json, json.dumps(report, indent=1))
    tables = os.path.join(results_dir, "tables")
    write_text(os.path.join(tables, f"retrieval_{args.split}.md"), markdown(report, used))
    write_text(os.path.join(tables, f"retrieval_{args.split}.tex"), latex(report, used, "all", MAIN_COLUMNS))
    write_text(os.path.join(tables, f"retrieval_{args.split}_full.tex"), latex(report, used, "all", FULL_COLUMNS))
    if "s200" in subsets:
        write_text(os.path.join(tables, f"retrieval_{args.split}_s200.tex"), latex(report, used, "s200", MAIN_COLUMNS))
    print(markdown(report, used))
    print(f"[retrieval] wrote {out_json} and tables in {tables}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
