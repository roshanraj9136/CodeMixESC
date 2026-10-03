"""Evaluates every available run (results/runs/{system}/{version}.jsonl) and writes the paper's
tables and figures. Metric definitions: codemixesc/metrics.py and docs/EVALUATION.md.

Each record is joined with its test sample by uid; reference, dialogue context and gold
strategy come from the test data, not from the run file. Every run is scored on three sets
of turns (scopes):
    all     every turn of the run (1,210 for the single-call baselines)
    subset  the run's turns among the fixed 200-turn subset sampled_uids(200)
    common  the turns of the subset that every system of the version answered; all comparison
            tables use it, so every row of a table is computed on identical turns.
A run that covers less than --min_coverage of the subset (e.g. still running) is scored but
left out of the common turns and the tables, so it cannot shrink everybody's turns.
cmx_nogate is derived from the codemixesc run: its pre-gate responses, without the gate's
LLM calls and latency.

Outputs under --out_dir (default results/):
    eval/metrics.json     every number (all scopes), run summaries and settings
    eval/per_turn.jsonl   per-turn scores of every run, for re-analysis without recomputation
    tables/{main_en,main_light,main_heavy,register,stability,ablation,efficiency,significance,
            baselines_all,runs,strategy_dist}.{md,csv,tex}
    figures/strategy_dist_{system}.png, figures/robustness.png (each with a vector .pdf twin)

Usage:
    python scripts/evaluate.py                                    # HingBERT-LID + BERTScore
    python scripts/evaluate.py --profiler lexicon --no_bertscore  # quick look, no model downloads
"""
import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc import metrics as M  # noqa: E402
from codemixesc.esconv import ROOT, all_samples, sampled_uids, seeker_utterances  # noqa: E402

SYSTEMS = ["zero_shot", "fewshot_cot", "maesc", "pivot", "codemixesc", "cmx_nogate", "cmx_noft", "cmx_noxl"]
MAIN_SYSTEMS = ["zero_shot", "fewshot_cot", "maesc", "pivot", "codemixesc"]
SINGLE_CALL = ("zero_shot", "fewshot_cot")
VERSIONS = ["en", "light", "heavy"]
REF_SYSTEM = "codemixesc"  # the system the significance tests compare against all others
DERIVED = {"cmx_nogate": "codemixesc"}
# The pivot runs MultiAgentESC on translations of the Hinglish turns, so its English
# counterpart for strategy stability is MultiAgentESC on the original English turns.
EN_COUNTERPART = {"pivot": "maesc"}
NAMES = {"zero_shot": "Zero-shot", "fewshot_cot": "Few-shot CoT", "maesc": "MultiAgentESC",
         "pivot": "Translate-Pivot", "codemixesc": "CodeMixESC", "cmx_nogate": "w/o Register Gate",
         "cmx_noft": "w/o retriever fine-tuning", "cmx_noxl": "w/o cross-lingual retriever",
         "reference": "Gold reference"}
VNAMES = {"en": "EN", "light": "Light", "heavy": "Heavy"}
PROFILER_NAMES = {"hingbert": "HingBERT-LID", "lexicon": "word list (approximate)", "none": "none"}
SIG_METRICS = [("f1", "F1", 2), ("rouge_l", "ROUGE-L", 2), ("chrf", "chrF", 2), ("bertscore", "BERTScore", 2),
               ("cmi_gap", "CMI gap", 3)]

# Figure palette (validated with the dataviz skill's checker: categorical slots in fixed order,
# one per system so a system keeps its colour in every figure; versions get an ordinal blue
# ramp, darker = more code-mixing).
SERIES_COLORS = {"codemixesc": "#2a78d6", "maesc": "#eb6834", "pivot": "#1baf7a", "fewshot_cot": "#eda100",
                 "zero_shot": "#e87ba4", "cmx_noft": "#008300", "cmx_noxl": "#4a3aa7", "cmx_nogate": "#e34948"}
MARKERS = {"codemixesc": "o", "maesc": "s", "pivot": "^", "fewshot_cot": "D", "zero_shot": "v",
           "cmx_noft": "P", "cmx_noxl": "X", "cmx_nogate": "*"}
VERSION_COLORS = {"en": "#86b6ef", "light": "#2a78d6", "heavy": "#104281"}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def safe_console():
    """Printing must not crash where the console encoding cannot show ×, ↓ or Devanagari
    (Windows code pages when output is redirected without -X utf8)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def say(msg):
    print(f"[eval] {msg}", flush=True)


def warn(msg):
    print(f"[eval] WARNING: {msg}", flush=True)


def name_of(system):
    return NAMES.get(system, system)


def uid_key(uid):
    parts = str(uid).split("-")
    return (0, tuple(int(p) for p in parts), "") if all(p.isdigit() for p in parts) else (1, (), str(uid))


def num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


# ================================================================== loading
def read_jsonl(path):
    """Records of a JSONL file; unreadable lines (a run killed mid-write) are counted, not fatal."""
    recs, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return recs, bad


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run(path):
    """{uid: record}. A later line replaces an earlier one with the same uid (a resumed run
    retries turns), except that a failed record never replaces a successful one."""
    raw, bad = read_jsonl(path)
    recs, dup = {}, 0
    for r in raw:
        uid = r.get("uid")
        if uid is None:
            bad += 1
            continue
        if uid in recs:
            dup += 1
            if "error" in r and "error" not in recs[uid]:
                continue
        recs[uid] = r
    return recs, {"duplicates": dup, "unreadable": bad}


def derive_nogate(recs):
    """cmx_nogate from codemixesc records (docs/RUN_FORMAT.md): response := pre_gate_response,
    n_calls -= gate.calls, latency -= gate.latency. Whether the gate's call was a cache hit is
    not recorded, so n_real_calls is not derivable and set to None."""
    out, fallback = {}, 0
    for uid, r in recs.items():
        g = r.get("gate") or {}
        d = dict(r, system="cmx_nogate", gate=None, n_real_calls=None)
        if "pre_gate_response" in r:
            d["response"] = r["pre_gate_response"]
        else:
            fallback += 1
        gc, gl = num(g.get("calls")) or 0.0, num(g.get("latency")) or 0.0
        if num(r.get("n_calls")) is not None:
            d["n_calls"] = r["n_calls"] - gc
        if num(r.get("latency")) is not None:
            d["latency"] = max(0.0, r["latency"] - gl)
        tags = dict(r.get("calls_by_tag") or {})
        if "gate" in tags:
            tags["gate"] -= gc
            if tags["gate"] <= 0:
                del tags["gate"]
        d["calls_by_tag"] = tags
        out[uid] = d
    if fallback:
        warn(f"{fallback} codemixesc records have no pre_gate_response; cmx_nogate uses their response")
    return out


def load_samples(versions, samples_file=None):
    """{version: {uid: sample}}; a version without test data is skipped with a warning."""
    if samples_file:
        data = read_json(samples_file)
        return {v: {s["uid"]: s for s in data[v]} for v in versions if v in data}
    out = {}
    for v in versions:
        try:
            out[v] = {s["uid"]: s for s in all_samples(v)}
        except FileNotFoundError as e:
            warn(f"no test data for version {v} ({e.filename}); version skipped")
    return out


def load_subset(subset_file, n):
    return set(read_json(subset_file)) if subset_file else set(sampled_uids(n))


def make_profiler(kind):
    if kind == "none":
        return None
    if kind == "lexicon":
        from codemixesc.testing import LexiconProfiler
        warn("--profiler lexicon: register metrics come from a word list (tests and dry runs only)")
        return LexiconProfiler()
    from codemixesc.profiler import Profiler
    try:
        return Profiler()
    except Exception as e:  # no network / model not cached
        sys.exit(f"[eval] HingBERT-LID could not be loaded ({e!r}). Use --profiler lexicon for a quick look "
                 "or --profiler none to skip the register metrics.")


def make_bertscore(args):
    if args.no_bertscore:
        return None
    try:
        return M.BERTScore(args.bertscore_model, batch_size=args.bertscore_batch)
    except Exception as e:
        warn(f"BERTScore unavailable ({e!r}); continuing without it (--no_bertscore silences this)")
        return None


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


# ================================================================== per-turn scores
def score_turns(recs, samples, ref_tokens):
    """Per-turn scores of one run (register and BERTScore are filled in later, in batches)."""
    turns = {}
    for uid, r in recs.items():
        s = samples[uid]
        missing = M.is_missing(r.get("response"))
        hyp, ref = M.hypothesis(r.get("response")), s["reference"].strip()
        if uid not in ref_tokens:
            ref_tokens[uid] = M.tokenize(ref)
        raw_pred = r.get("pred_strategy")
        pred = M.normalize_strategy(raw_pred)
        gate = r.get("gate") if isinstance(r.get("gate"), dict) else None
        turns[uid] = {
            "uid": uid, "early": s.get("early"), "failed": "error" in r, "missing": missing,
            "hyp": hyp, "ref": ref, "hyp_tokens": M.tokenize(hyp), "ref_tokens": ref_tokens[uid],
            "f1": M.unigram_f1(hyp, ref), "rouge_l": M.rouge_l(hyp, ref), "chrf": M.chrf_sentence(hyp, ref),
            "bertscore": None, "register": None, "cmi_gap": None,
            "words": None if missing else M.word_count(hyp),
            "pred": pred, "invalid_strategy": pred == "None" and str(raw_pred or "None").strip() != "None",
            "gold": M.gold_strategies(s.get("strategy")), "gold_raw": s.get("strategy"),
            "path": r.get("path"), "n_calls": num(r.get("n_calls")), "n_real_calls": num(r.get("n_real_calls")),
            "n_failed_calls": num(r.get("n_failed_calls")), "n_truncated_calls": num(r.get("n_truncated_calls")),
            "latency": num(r.get("latency")), "calls_by_tag": r.get("calls_by_tag") or {},
            "gate_triggered": gate.get("triggered") if gate else None,
            "gate_accepted": gate.get("accepted") if gate else None,
            "stale": str(r.get("reference", s["reference"]) or "").strip() != ref
                     or str(r.get("post", s["post"]) or "").strip() != str(s["post"] or "").strip(),
        }
    return turns


def add_register(turns_by_run, samples, prof):
    """Register match of every produced response; one batched tagging pass over all texts."""
    texts = set()
    for (s, v), turns in turns_by_run.items():
        for uid, t in turns.items():
            texts.update(seeker_utterances(samples[v][uid]))
            texts.add(t["ref"])
            if not t["missing"]:
                texts.add(t["hyp"])
    prof.tag_many(sorted(texts))  # profile() and stats() below then hit the profiler's memo
    for (s, v), turns in turns_by_run.items():
        for uid, t in turns.items():
            if not t["missing"]:
                t["register"] = M.register_match(prof, seeker_utterances(samples[v][uid]), t["hyp"])
                t["cmi_gap"] = t["register"]["cmi_gap"]


def add_bertscore(turns_by_run, bert):
    pairs = sorted({(t["hyp"], t["ref"]) for turns in turns_by_run.values() for t in turns.values() if t["hyp"]})
    say(f"BERTScore ({bert.model_type}, {bert.device}) on {len(pairs)} distinct response-reference pairs")
    scores = dict(zip(pairs, bert([h for h, _ in pairs], [r for _, r in pairs])))
    for turns in turns_by_run.values():
        for t in turns.values():
            t["bertscore"] = scores[(t["hyp"], t["ref"])] if t["hyp"] else 0.0


# ================================================================== aggregation
def aggregate(turns, maesc=None):
    """Every metric over a list of per-turn dicts; maesc = MultiAgentESC's turns of the same
    version, for the extra calls over the base system on shared turns."""
    out = {"n": len(turns)}
    if not turns:
        return out
    hyp_toks = [t["hyp_tokens"] for t in turns]
    produced = [t for t in turns if not t["missing"]]
    out.update(
        n_failed=sum(t["failed"] for t in turns), n_none=sum(t["missing"] and not t["failed"] for t in turns),
        distinct_1=M.distinct_n(hyp_toks, 1), distinct_2=M.distinct_n(hyp_toks, 2),
        bleu_1=M.corpus_bleu(hyp_toks, [t["ref_tokens"] for t in turns], 1),
        bleu_2=M.corpus_bleu(hyp_toks, [t["ref_tokens"] for t in turns], 2),
        bleu_3=M.corpus_bleu(hyp_toks, [t["ref_tokens"] for t in turns], 3),
        f1=M.mean(t["f1"] for t in turns), rouge_l=M.mean(t["rouge_l"] for t in turns),
        chrf=M.chrf_corpus([t["hyp"] for t in turns], [t["ref"] for t in turns]),
        chrf_sentence=M.mean(t["chrf"] for t in turns),
        bertscore_f1=None if any(t["bertscore"] is None for t in turns) else M.mean(t["bertscore"] for t in turns),
    )
    if any(t["register"] is not None for t in turns):
        out.update(M.register_summary(t["register"] for t in turns))
    preds, golds = [t["pred"] for t in turns], [t["gold"] for t in turns]
    out["strategy_match"], out["strategy_match_predicted"] = M.strategy_match(preds, golds)
    out["strategy_counts"] = dict(zip(M.LABELS, M.strategy_counts(preds)))
    out["n_invalid_strategy"] = sum(t["invalid_strategy"] for t in turns)
    out.update(
        n_calls=M.mean(t["n_calls"] for t in turns), n_real_calls=M.mean(t["n_real_calls"] for t in turns),
        latency=M.mean(t["latency"] for t in turns),
        words=M.mean(t["words"] for t in produced), over_30_words=M.pct(t["words"] > 30 for t in produced),
        paths=dict(Counter(t["path"] or "unknown" for t in turns)),
        multi_path=M.pct(t["path"] == "multi" for t in turns),
        calls_by_tag={k: v / len(turns) for k, v in sorted(sum((Counter(t["calls_by_tag"]) for t in turns),
                                                                Counter()).items())},
    )
    gated = [t for t in turns if t["gate_triggered"] is not None]
    if gated:  # % of turns where the gate fired, and % of those where its rewrite was kept
        out["gate_triggered"] = M.pct(t["gate_triggered"] for t in gated)
        out["gate_accepted"] = M.pct(t["gate_accepted"] for t in gated if t["gate_triggered"])
    if maesc is not None:
        shared = [(t, maesc[t["uid"]]) for t in turns if t["uid"] in maesc]
        calls = [(a["n_calls"], b["n_calls"]) for a, b in shared if None not in (a["n_calls"], b["n_calls"])]
        lat = [(a["latency"], b["latency"]) for a, b in shared if None not in (a["latency"], b["latency"])]
        out["n_vs_maesc"] = len(calls)
        out["extra_calls_vs_maesc"] = M.mean(a - b for a, b in calls)
        out["extra_latency_vs_maesc"] = M.mean(a - b for a, b in lat)
    return out


def call_total(turns, field):
    """Sum of a per-turn call counter over a run; None for runs written before the field existed."""
    vals = [t[field] for t in turns.values() if t[field] is not None]
    return int(sum(vals)) if vals else None


def scope_sets(turns, subset, common_v, in_tables):
    uids = set(turns)
    sets = {"all": uids, "subset": uids & subset}
    if in_tables:
        sets["common"] = set(common_v)
    return sets


# ================================================================== tables
class Col:
    """A table column. fmt None = text; best 'max'/'min' = bold the best value per block;
    merge = show a value only where it changes within a block (e.g. the version)."""

    def __init__(self, key, md, tex=None, fmt=".2f", best=None, group=None, merge=False):
        self.key, self.md, self.tex, self.fmt = key, md, tex if tex is not None else tex_escape(md), fmt
        self.best, self.group, self.merge = best, group, merge


_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                # OT1-encoded text fonts (IEEEtran's default) print | < > as other glyphs
                "|": "$|$", "<": "$<$", ">": "$>$",
                "×": r"$\times$", "↓": r"$\downarrow$", "↑": r"$\uparrow$", "→": r"$\rightarrow$", "−": "$-$",
                "–": "--", "†": r"$^{\dagger}$", "‡": r"$^{\ddagger}$", "δ": r"$\delta$", "Δ": r"$\Delta$",
                "≥": r"$\geq$"}


_TEX_PHRASES = {"|CMI_r − CMI_s|": r"$|\mathrm{CMI}_r - \mathrm{CMI}_s|$", "CMI_r": r"CMI$_r$", "CMI_s": r"CMI$_s$"}
_TEX_PHRASE_RE = re.compile("|".join(re.escape(p) for p in sorted(_TEX_PHRASES, key=len, reverse=True)))


def tex_escape(text):
    """Plain text (captions, names) as LaTeX that compiles with pdflatex in any font encoding."""
    text, out, pos = str(text), [], 0
    for m in _TEX_PHRASE_RE.finditer(text):
        out += [_TEX_SPECIAL.get(ch, ch) for ch in text[pos:m.start()]] + [_TEX_PHRASES[m.group(0)]]
        pos = m.end()
    return "".join(out + [_TEX_SPECIAL.get(ch, ch) for ch in text[pos:]])


def tex_number(text):
    return "$-$" + text[1:] if text.startswith("-") else ("$+$" + text[1:] if text.startswith("+") else text)


def _rounded(value, fmt):
    try:
        return float(format(value, fmt))
    except (TypeError, ValueError):
        return None


def write_table(tables_dir, name, cols, rows, caption, note=None):
    """Writes tables/{name}.md, .csv and .tex. rows are dicts with the column keys, plus
    '_block' (best values are marked per block; a rule separates blocks) and '_nobold'.
    A value may be a dict {'md', 'tex', 'csv'} for pre-rendered cells."""
    if not rows:
        say(f"table {name}: no data, not written")
        return
    cols = [c for c in cols if c.fmt is None or any(r.get(c.key) is not None for r in rows)]
    blocks = []
    for r in rows:
        if not blocks or blocks[-1][0] != r.get("_block"):
            blocks.append((r.get("_block"), []))
        blocks[-1][1].append(r)
    bold = set()
    for _, brows in blocks:
        for c in cols:
            if c.best not in ("max", "min"):
                continue
            vals = [(i, _rounded(r.get(c.key), c.fmt)) for i, r in enumerate(brows)
                    if not r.get("_nobold") and isinstance(r.get(c.key), (int, float))]
            vals = [(i, v) for i, v in vals if v is not None]
            if len(vals) < 2:
                continue
            best = (max if c.best == "max" else min)(v for _, v in vals)
            if all(v == best for _, v in vals):
                continue  # a tie of every row marks nothing
            bold.update((id(brows[i]), c.key) for i, v in vals if v == best)

    def cell(r, c, prev, kind):
        v = r.get(c.key)
        if c.merge and prev is not None and prev.get(c.key) == v:
            return ""
        if v is None:
            return "–" if kind == "md" else "--"
        if isinstance(v, dict):
            return v[kind]
        if c.fmt is None:
            return str(v) if kind == "md" else tex_escape(v)
        text = format(v, c.fmt)
        if kind == "tex":
            text = tex_number(text)
        if (id(r), c.key) in bold:
            return f"**{text}**" if kind == "md" else f"\\textbf{{{text}}}"
        return text

    os.makedirs(tables_dir, exist_ok=True)
    # Markdown
    lines = [f"**{caption}**", "", "| " + " | ".join((f"{c.group} {c.md}" if c.group else c.md) for c in cols) + " |",
             "|" + "|".join(":--" if c.fmt is None else "--:" for c in cols) + "|"]
    for _, brows in blocks:
        prev = None
        for r in brows:
            lines.append("| " + " | ".join(cell(r, c, prev, "md") for c in cols) + " |")
            prev = r
    if note:
        lines += ["", note]
    with open(os.path.join(tables_dir, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    # CSV: raw numbers, a pre-rendered cell's 'csv' dict becomes one column per entry
    header, getters = [], []
    for c in cols:
        key = c.key
        subs = next((list(r[c.key]["csv"]) for r in rows if isinstance(r.get(c.key), dict)
                     and isinstance(r[c.key].get("csv"), dict)), None)
        if subs:
            for sk in subs:
                header.append(f"{key}_{sk}")
                getters.append(lambda r, c=c, sk=sk: (r.get(c.key) or {}).get("csv", {}).get(sk))
        else:
            header.append(key)
            getters.append(lambda r, c=c: (r.get(c.key) or {}).get("csv") if isinstance(r.get(c.key), dict)
                           else r.get(c.key))
    with open(os.path.join(tables_dir, f"{name}.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if g(r) is None else (round(g(r), 6) if isinstance(g(r), float) else g(r))
                        for g in getters])
    # LaTeX (booktabs; shrinks to the line width only when wider)
    spec = "".join("l" if c.fmt is None else "r" for c in cols)
    env = "table*" if len(cols) >= 10 else "table"  # wide tables span both columns of a two-column paper
    tex = ["% Generated by scripts/evaluate.py -- do not edit by hand. Needs \\usepackage{booktabs,graphicx}.",
           f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption_tex(caption, note)}}}",
           f"\\label{{tab:{name}}}",
           "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
           f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    if any(c.group for c in cols):
        heads, rules, i = [], [], 0
        while i < len(cols):
            j = i
            while j + 1 < len(cols) and cols[j + 1].group == cols[i].group:
                j += 1
            g = cols[i].group
            if g:
                heads.append(f"\\multicolumn{{{j - i + 1}}}{{c}}{{{tex_escape(g)}}}")
                rules.append(f"\\cmidrule(lr){{{i + 1}-{j + 1}}}")
            else:
                heads += [""] * (j - i + 1)
            i = j + 1
        tex += [" & ".join(heads) + " \\\\", " ".join(rules)]
    tex.append(" & ".join(c.tex for c in cols) + " \\\\")
    tex.append("\\midrule")
    for bi, (_, brows) in enumerate(blocks):
        if bi:
            tex.append("\\midrule")
        prev = None
        for r in brows:
            tex.append(" & ".join(cell(r, c, prev, "tex") for c in cols) + " \\\\")
            prev = r
    tex += ["\\bottomrule", "\\end{tabular}}", f"\\end{{{env}}}"]
    with open(os.path.join(tables_dir, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


def caption_tex(caption, note=None):
    return tex_escape(caption + (" " + note if note else ""))


QUALITY_COLS = [Col("distinct_1", "D-1", best="max"), Col("distinct_2", "D-2", best="max"),
                Col("bleu_1", "B-1", best="max"), Col("bleu_2", "B-2", best="max"), Col("bleu_3", "B-3", best="max"),
                Col("f1", "F1", best="max"), Col("rouge_l", "R-L", best="max"), Col("chrf", "chrF", best="max"),
                Col("bertscore_f1", "BERTScore", best="max")]


def quality_note(with_bertscore):
    return ("Distinct-n (D), corpus BLEU-n (B), unigram F1, ROUGE-L (R-L), corpus chrF"
            + (" and multilingual BERTScore F1" if with_bertscore else "") + ", all ×100; best per column in bold.")


def sig_cell(res, digits):
    if not res:
        return None
    d, lo, hi, p = res["diff"], res["ci_low"], res["ci_high"], res["p"]
    md_mark = "‡" if p < 0.01 else ("†" if p < 0.05 else "")
    tex_mark = r"$^{\ddagger}$" if p < 0.01 else (r"$^{\dagger}$" if p < 0.05 else "")
    return {"md": f"{d:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]{md_mark}",
            "tex": tex_number(f"{d:+.{digits}f}") + tex_mark,
            "csv": {"diff": d, "ci_low": lo, "ci_high": hi, "p": p}}


TABLE_NAMES = ["main_en", "main_light", "main_heavy", "register", "stability", "ablation", "efficiency",
               "significance", "baselines_all", "runs", "strategy_dist"]


def remove_stale(paths):
    """Outputs of an earlier evaluation that this one does not rewrite would otherwise sit next to
    a metrics.json they do not match; only files this script generates are removed."""
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def write_tables(tables_dir, ctx):
    remove_stale(os.path.join(tables_dir, f"{n}.{ext}") for n in TABLE_NAMES for ext in ("md", "csv", "tex"))
    R, common, versions, in_tables = ctx["results"], ctx["common"], ctx["versions"], ctx["in_tables"]
    ordered = lambda v, pool: [s for s in pool if s in in_tables.get(v, ())]  # noqa: E731
    others = [s for s in ctx["systems"] if s not in MAIN_SYSTEMS]  # ablations and custom variants

    def res(v, s, scope="common"):
        return (R.get(v, {}).get(s) or {}).get(scope) or {}

    # main tables, one per version
    for v in versions:
        rows = [dict(res(v, s), system=name_of(s)) for s in ordered(v, MAIN_SYSTEMS)]
        write_table(tables_dir, f"main_{v}", [Col("system", "System", fmt=None)] + QUALITY_COLS, rows,
                    f"Response quality on {VNAMES[v]} ({len(common.get(v, ()))} common turns of the "
                    f"{ctx['subset_size']}-turn subset).",
                    quality_note(any(r.get("bertscore_f1") is not None for r in rows)))

    # register match
    reg_versions = [v for v in versions if any(res(v, s).get("cmi_gap") is not None for s in in_tables.get(v, ()))]
    reg_cols = [Col("system", "System", fmt=None)]
    for v in reg_versions:
        reg_cols += [Col(f"{v}_cmi_gap", "CMI gap ↓", "CMI gap $\\downarrow$", ".3f", "min", VNAMES[v]),
                     Col(f"{v}_hi_frac_gap", "Hi-frac gap ↓", "Hi gap $\\downarrow$", ".3f", "min", VNAMES[v]),
                     Col(f"{v}_script", "Script % ↑", "Script $\\uparrow$", ".1f", "max", VNAMES[v])]
    rows = []
    for s in [s for s in MAIN_SYSTEMS + others if any(s in in_tables.get(v, ()) for v in reg_versions)]:
        row = {"system": name_of(s)}
        for v in reg_versions:
            r = res(v, s)
            row.update({f"{v}_cmi_gap": r.get("cmi_gap"), f"{v}_hi_frac_gap": r.get("hi_frac_gap"),
                        f"{v}_script": r.get("script_consistency")})
        rows.append(row)
    if rows:
        ref = {"system": name_of("reference"), "_nobold": True}
        for v in reg_versions:
            g = ctx["reference_register"].get(v) or {}
            ref.update({f"{v}_cmi_gap": g.get("cmi_gap"), f"{v}_hi_frac_gap": g.get("hi_frac_gap"),
                        f"{v}_script": g.get("script_consistency")})
        rows.append(ref)
    seeker = ", ".join(f"{VNAMES[v]} {ctx['reference_register'][v]['cmi_s']:.3f}" for v in reg_versions
                       if (ctx["reference_register"].get(v) or {}).get("cmi_s") is not None)
    write_table(tables_dir, "register", reg_cols, rows,
                "Register match of the responses to the seeker on the common turns: mean CMI gap |CMI_r − CMI_s| "
                "and Hindi-fraction gap (lower is better), and script consistency in % (higher is better).",
                f"Mean seeker CMI_s: {seeker}. Gold reference: the dataset's own supporter turns. Profiler: "
                f"{ctx['profiler']}.")

    # strategy stability (EN vs each Hinglish version, same turns)
    st_versions = [v for v in versions if v != "en" and ctx["stability"].get(v)]
    st_cols = [Col("system", "System", fmt=None)]
    for v in st_versions:
        st_cols += [Col(f"{v}_jsd", "JSD ↓", "JSD $\\downarrow$", ".3f", "min", f"EN→{VNAMES[v]}"),
                    Col(f"{v}_jsd_s", "JSD-S ↓", "JSD-S $\\downarrow$", ".3f", "min", f"EN→{VNAMES[v]}"),
                    Col(f"{v}_agree", "Agree % ↑", "Agr. $\\uparrow$", ".1f", "max", f"EN→{VNAMES[v]}")]
    rows = []
    for s in [s for s in MAIN_SYSTEMS + others if any(s in ctx["stability"].get(v, {}) for v in st_versions)]:
        row = {"system": name_of(s) + (" †" if s in EN_COUNTERPART else "")}
        for v in st_versions:
            r = (ctx["stability"][v].get(s) or {}).get("common") or {}
            row.update({f"{v}_jsd": r.get("jsd"), f"{v}_jsd_s": r.get("jsd_strategies"),
                        f"{v}_agree": r.get("agreement")})
        rows.append(row)
    write_table(tables_dir, "stability", st_cols, rows,
                "Strategy stability under code-mixing on the common turns: Jensen–Shannon divergence (bits) between "
                "the predicted-strategy distributions of EN and of the Hinglish version, over the 8 strategies + "
                "None (JSD) and over the 8 strategies on turns where both versions predicted one (JSD-S), and "
                "per-turn agreement in %.",
                "† compared with MultiAgentESC on EN (the pivot runs MultiAgentESC on translations).")

    # ablations, one block per version
    abl = [s for s in [REF_SYSTEM] + others]
    rows = []
    for v in versions:
        present = [s for s in abl if s in in_tables.get(v, ())]
        if len(present) < 2 or not any(s in present for s in others):
            continue
        for s in present:
            r = res(v, s)
            rows.append(dict(r, version=VNAMES[v], system=name_of(s), script=r.get("script_consistency"), _block=v))
    write_table(tables_dir, "ablation",
                [Col("version", "Version", fmt=None, merge=True), Col("system", "System", fmt=None),
                 Col("distinct_2", "D-2", best="max"), Col("bleu_2", "B-2", best="max"), Col("f1", "F1", best="max"),
                 Col("rouge_l", "R-L", best="max"), Col("chrf", "chrF", best="max"),
                 Col("bertscore_f1", "BERTScore", best="max"),
                 Col("cmi_gap", "CMI gap ↓", "CMI gap $\\downarrow$", ".3f", "min"),
                 Col("script", "Script % ↑", "Script $\\uparrow$", ".1f", "max")],
                rows, "Ablations of CodeMixESC on the common turns.",
                "Quality scores ×100; best per version in bold. w/o Register Gate is the same run before the gate.")

    # efficiency, one block per version
    rows = []
    for v in versions:
        for s in ordered(v, MAIN_SYSTEMS + others):
            r = res(v, s)
            rows.append(dict(r, version=VNAMES[v], system=name_of(s), _block=v))
    write_table(tables_dir, "efficiency",
                [Col("version", "Version", fmt=None, merge=True), Col("system", "System", fmt=None),
                 Col("n_calls", "Calls/turn", best="min", fmt=".2f"),
                 Col("extra_calls_vs_maesc", "Δ calls vs MAESC", "$\\Delta$ calls", fmt="+.2f"),
                 Col("latency", "Latency (s)", fmt=".1f", best="min"),
                 Col("multi_path", "Multi-path %", "Multi \\%", fmt=".1f"),
                 Col("words", "Words", fmt=".1f"),
                 Col("over_30_words", ">30 words %", "$>$30 w. \\%", fmt=".1f", best="min")],
                rows, "Efficiency on the common turns: logical LLM calls and summed API latency per turn, extra calls "
                      "over MultiAgentESC on the same turns, share of turns on the full multi-agent path, and "
                      "response length.",
                "Calls count cached calls too; latency is the API time of the original calls.")

    # significance: REF_SYSTEM against every other system of the version
    rows = []
    for v in versions:
        for s, entry in (ctx["significance"].get(v) or {}).items():
            row = {"version": VNAMES[v], "comparison": f"vs {name_of(s)}", "_block": v,
                   "n": max((e["n"] for e in entry.values()), default=None)}
            for key, _, digits in SIG_METRICS:
                row[key] = sig_cell(entry.get(key), digits)
            rows.append(row)
    write_table(tables_dir, "significance",
                [Col("version", "Version", fmt=None, merge=True), Col("comparison", name_of(REF_SYSTEM), fmt=None),
                 Col("n", "n", fmt="d")] + [Col(k, label, fmt=".2f") for k, label, _ in SIG_METRICS],
                rows, f"Paired bootstrap ({ctx['n_boot']} resamples, seed {ctx['seed']}) of the per-turn score "
                      f"differences Δ = {name_of(REF_SYSTEM)} − system on the common turns (sentence chrF; CMI gap: "
                      "lower is better).",
                "† p < 0.05, ‡ p < 0.01 (two-sided); the Markdown and CSV versions give the 95% CIs.")

    # single-call baselines on all turns vs the subset (is the subset representative?)
    rows = []
    for v in versions:
        for s in SINGLE_CALL:
            for scope, label in (("all", "all turns"), ("subset", "subset")):
                r = res(v, s, scope)
                if r.get("n"):
                    rows.append(dict(r, version=VNAMES[v], system=f"{name_of(s)} ({label})", _block=v))
    write_table(tables_dir, "baselines_all",
                [Col("version", "Version", fmt=None, merge=True), Col("system", "System", fmt=None),
                 Col("n", "Turns", fmt="d"), Col("distinct_1", "D-1"), Col("distinct_2", "D-2"),
                 Col("bleu_2", "B-2"), Col("f1", "F1"), Col("rouge_l", "R-L"), Col("chrf", "chrF"),
                 Col("bertscore_f1", "BERTScore"), Col("cmi_gap", "CMI gap", fmt=".3f")],
                rows, "Single-call baselines on all test turns and on the sampled subset (representativeness check).")

    # runs
    rows = []
    for key, sm in ctx["runs"].items():
        rows.append({"system": sm["system"], "version": VNAMES.get(sm["version"], sm["version"]),
                     "records": sm["records"], "coverage": sm["subset_coverage"], "missing": sm["missing"],
                     "failed": sm["failed"], "none": sm["none"], "duplicates": sm["duplicates"],
                     "failed_calls": sm["failed_calls"], "truncated_calls": sm["truncated_calls"],
                     "unreadable": sm["unreadable"], "stale": sm["stale"], "in_tables": "yes" if sm["in_tables"] else "no",
                     "model": (sm.get("meta") or {}).get("model"), "delta": (sm.get("meta") or {}).get("delta")})
    write_table(tables_dir, "runs",
                [Col("system", "System", fmt=None), Col("version", "Version", fmt=None),
                 Col("records", "Records", fmt="d"), Col("coverage", "Subset %", "Subset \\%", fmt=".1f"),
                 Col("missing", "Missing", fmt="d"), Col("failed", "Failed", fmt="d"), Col("none", "None", fmt="d"),
                 Col("failed_calls", "Empty calls", fmt="d"), Col("truncated_calls", "Truncated calls", fmt="d"),
                 Col("duplicates", "Dup.", fmt="d"), Col("unreadable", "Unread.", fmt="d"),
                 Col("stale", "Stale", fmt="d"),
                 Col("in_tables", "In tables", fmt=None), Col("model", "Model", fmt=None),
                 Col("delta", "δ", "$\\delta$", fmt=".2f")],
                rows, "Runs: records, coverage of the sampled subset, missing turns (expected but absent), failed "
                      "turns (error field), 'None' responses, LLM calls that returned an empty or blocked answer "
                      "and calls cut at the token limit (totals), duplicate uids, unreadable lines, and records "
                      "whose reference or post differs from the current test data (stale).")

    # strategy distributions (the table view of the strategy figures)
    rows = []
    for v in versions:
        for s in ordered(v, MAIN_SYSTEMS + others):
            counts = res(v, s).get("strategy_counts")
            if counts and sum(counts.values()) and any(counts[k] for k in M.STRATEGIES):
                n = sum(counts.values())
                rows.append(dict({lab: 100.0 * counts[lab] / n for lab in M.LABELS}, version=VNAMES[v],
                                 system=name_of(s), n=n, _block=v))
    write_table(tables_dir, "strategy_dist",
                [Col("version", "Version", fmt=None, merge=True), Col("system", "System", fmt=None),
                 Col("n", "Turns", fmt="d")] + [Col(lab, lab, tex_escape(lab), fmt=".1f") for lab in M.LABELS],
                rows, "Predicted-strategy distribution (% of the common turns).")


# ================================================================== figures
def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=7, length=0, pad=3)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight", facecolor="white")


def plot_strategy_dist(path, title, series):
    """series: list of (label, version, counts over M.LABELS). Grouped horizontal bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    k, y = len(series), np.arange(len(M.LABELS))
    h = 0.8 / k
    for i, (label, v, counts) in enumerate(series):
        share = 100.0 * np.asarray(counts, dtype=float) / max(1, sum(counts))
        ax.barh(y - 0.4 + h * (i + 0.5), share, height=h, color=VERSION_COLORS[v], edgecolor="white",
                linewidth=0.6, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(M.LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Share of turns (%)", fontsize=7, color=INK2)
    ax.set_title(title, fontsize=8, color=INK, loc="left", pad=16)
    _style(ax, grid_axis="x")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(series),
              labelcolor=INK2, borderaxespad=0.2, handlelength=1.2, columnspacing=1.0)
    _save(fig, path)
    plt.close(fig)


def plot_robustness(path, results, versions, systems, in_tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("f1", "F1"), ("rouge_l", "ROUGE-L"), ("chrf", "chrF"), ("bertscore_f1", "BERTScore F1"),
               ("cmi_gap", "CMI gap (lower is better)")]

    def val(v, s, key):
        return (((results.get(v) or {}).get(s) or {}).get("common") or {}).get(key) if s in in_tables.get(v, ()) \
            else None

    systems = [s for s in systems if sum(val(v, s, "f1") is not None for v in versions) >= 2]
    metrics = [(k, n) for k, n in metrics if any(val(v, s, k) is not None for v in versions for s in systems)]
    if not systems or not metrics:
        return False
    fig, axes = plt.subplots(1, len(metrics), figsize=(1.55 * len(metrics) + 0.5, 2.2), squeeze=False)
    for ax, (key, label) in zip(axes[0], metrics):
        for s in systems:
            pts = [(i, val(v, s, key)) for i, v in enumerate(versions) if val(v, s, key) is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES_COLORS.get(s, "#898781"),
                        marker=MARKERS.get(s, "o"), linewidth=1.5, markersize=5, markeredgecolor="white",
                        markeredgewidth=0.8, label=name_of(s), solid_capstyle="round")
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels([VNAMES[v] for v in versions])
        ax.set_xlim(-0.3, len(versions) - 0.7)
        ax.set_title(label, fontsize=7.5, color=INK)
        _style(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.08), labelcolor=INK2)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)
    return True


def write_figures(fig_dir, ctx):
    os.makedirs(fig_dir, exist_ok=True)
    remove_stale(os.path.join(fig_dir, f) for f in os.listdir(fig_dir)
                 if re.fullmatch(r"(strategy_dist_\w+|robustness)\.(png|pdf)", f))
    versions, turns, common, in_tables = ctx["versions"], ctx["turns"], ctx["common"], ctx["in_tables"]
    for s in ctx["systems"]:
        if s in DERIVED:
            continue  # same strategies as its source run
        vs = [v for v in versions if s in in_tables.get(v, ())]
        en_sys = None
        if "en" in versions and "en" not in vs and EN_COUNTERPART.get(s) in in_tables.get("en", ()):
            en_sys = EN_COUNTERPART[s]
        if not vs:
            continue
        uids = set.intersection(*[set(common[v]) for v in vs + (["en"] if en_sys else [])])
        series = []
        if en_sys:
            series.append((f"EN ({name_of(en_sys)})", "en",
                           M.strategy_counts([turns[(en_sys, "en")][u]["pred"] for u in uids])))
        for v in vs:
            series.append((VNAMES[v], v, M.strategy_counts([turns[(s, v)][u]["pred"] for u in uids])))
        if not uids or not any(sum(c[:len(M.STRATEGIES)]) for _, _, c in series):
            continue  # never predicts a strategy (zero-shot)
        plot_strategy_dist(os.path.join(fig_dir, f"strategy_dist_{s}.png"),
                           f"{name_of(s)}: predicted strategies ({len(uids)} turns)", series)
    plot_robustness(os.path.join(fig_dir, "robustness.png"), ctx["results"], versions,
                    [s for s in ["codemixesc", "maesc", "pivot", "fewshot_cot", "zero_shot"] if s in ctx["systems"]],
                    in_tables)


# ================================================================== output helpers
def clean(obj):
    """JSON-safe copy: sets -> sorted lists, floats rounded, NaN -> None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, set):
        return sorted(clean(v) for v in obj)
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    try:
        f = float(obj)
    except (TypeError, ValueError):
        return str(obj)
    return None if math.isnan(f) or math.isinf(f) else round(f, 6)


PER_TURN_FIELDS = ["early", "path", "failed", "missing", "pred", "gold_raw", "f1", "rouge_l", "chrf", "bertscore",
                   "words", "n_calls", "latency", "gate_triggered", "gate_accepted"]


def write_per_turn(path, turns_by_run):
    with open(path, "w", encoding="utf-8") as f:
        for (s, v) in sorted(turns_by_run, key=lambda k: (SYSTEMS.index(k[0]) if k[0] in SYSTEMS else 99, k[0],
                                                          VERSIONS.index(k[1]))):
            for uid in sorted(turns_by_run[(s, v)], key=uid_key):
                t = turns_by_run[(s, v)][uid]
                rec = {"system": s, "version": v, "uid": uid, **{k: t[k] for k in PER_TURN_FIELDS}}
                reg = t["register"] or {}
                rec.update({k: reg.get(k) for k in ("cmi_s", "cmi_r", "cmi_gap", "hi_frac_s", "hi_frac_r",
                                                     "hi_frac_gap", "script_s", "script_r")})
                f.write(json.dumps(clean(rec), ensure_ascii=False) + "\n")


# ================================================================== main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results"),
                    help="writes eval/, tables/ and figures/ below it")
    ap.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated; cmx_nogate is derived")
    ap.add_argument("--versions", default=",".join(VERSIONS))
    ap.add_argument("--profiler", choices=["hingbert", "lexicon", "none"], default="hingbert",
                    help="register metrics: HingBERT-LID (reported results), a word list (tests, dry runs), or none")
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--bertscore_model", default=M.BERTSCORE_MODEL)
    ap.add_argument("--bertscore_batch", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset_size", type=int, default=200)
    ap.add_argument("--min_coverage", type=float, default=0.9,
                    help="runs covering less of the subset are scored but kept out of the common turns and tables")
    ap.add_argument("--hien_dir", help="ESConv-HiEn directory (default: $CODEMIX_HIEN_DIR or data/esconv_hien)")
    ap.add_argument("--samples_file", help="JSON {version: [sample, ...]} instead of the test data (tests)")
    ap.add_argument("--subset_file", help="JSON list of uids instead of sampled_uids(subset_size) (tests)")
    ap.add_argument("--no_figures", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    safe_console()
    if args.hien_dir:
        os.environ["CODEMIX_HIEN_DIR"] = os.path.abspath(args.hien_dir)
    systems = [s for s in args.systems.split(",") if s]
    versions = [v for v in VERSIONS if v in args.versions.split(",")]
    samples = load_samples(versions, args.samples_file)
    versions = [v for v in versions if v in samples]
    if not versions:
        sys.exit("[eval] no test data for any requested version")
    subset = load_subset(args.subset_file, args.subset_size)

    # ---- runs
    recs, runs = {}, {}
    for s in systems:
        for v in versions:
            if s in DERIVED:
                continue
            path = os.path.join(args.runs_dir, s, f"{v}.jsonl")
            if not os.path.exists(path):
                continue
            r, info = load_run(path)
            unknown = [u for u in r if u not in samples[v]]
            for u in unknown:
                del r[u]
            if not r:
                warn(f"{s}/{v}: no usable records in {path}")
                continue
            mismatch = sum(1 for x in r.values() if x.get("version", v) != v or x.get("system", s) != s)
            if mismatch:
                warn(f"{s}/{v}: {mismatch} records name another system or version")
            meta_path = os.path.join(args.runs_dir, s, f"{v}.meta.json")
            err_path = os.path.join(args.runs_dir, s, f"{v}.errors.jsonl")
            meta = read_json(meta_path) if os.path.exists(meta_path) else None
            errors = {e.get("uid") for e in read_jsonl(err_path)[0]} - set(r) if os.path.exists(err_path) else set()
            recs[(s, v)] = r
            runs[(s, v)] = dict(info, unknown_uid=len(unknown), errors_unresolved=len(errors), path=path,
                                meta={k: meta.get(k) for k in ("model", "delta", "git_commit", "finished", "spec")}
                                if meta else None)
    ablation_runs = [s for s in systems if s not in MAIN_SYSTEMS and s not in DERIVED]
    for s, src in DERIVED.items():
        for v in versions:  # ablations are run on the Hinglish versions; EN only if an EN ablation exists
            if v == "en" and not any((a, "en") in recs for a in ablation_runs):
                continue
            if s in systems and (src, v) in recs:
                recs[(s, v)] = derive_nogate(recs[(src, v)])
                runs[(s, v)] = dict(runs[(src, v)], derived_from=src)
    if not recs:
        sys.exit(f"[eval] no runs found in {args.runs_dir}")
    say(f"{len(recs)} runs: " + ", ".join(f"{s}/{v}" for s, v in recs))

    # ---- per-turn scores
    ref_tokens = {v: {} for v in versions}
    turns = {(s, v): score_turns(r, samples[v], ref_tokens[v]) for (s, v), r in recs.items()}
    prof = make_profiler(args.profiler)
    if prof is not None:
        say("register profiles ...")
        add_register(turns, samples, prof)
    bert = make_bertscore(args)
    if bert is not None:
        add_bertscore(turns, bert)

    # ---- which runs enter the tables, and the common turns of each version
    in_tables, common = {}, {}
    for v in versions:
        present = [s for s in systems if (s, v) in turns]
        cov = {s: len(set(turns[(s, v)]) & subset) / max(1, len(subset)) for s in present}
        in_tables[v] = [s for s in present if cov[s] >= args.min_coverage]
        for s in present:
            if s not in in_tables[v]:
                warn(f"{s}/{v} covers {100 * cov[s]:.0f}% of the subset (< {100 * args.min_coverage:.0f}%): "
                     "scored, but left out of the common turns and the tables")
        common[v] = set.intersection(*[set(turns[(s, v)]) & subset for s in in_tables[v]]) if in_tables[v] else set()
        if in_tables[v] and len(common[v]) < len(subset):
            warn(f"{v}: {len(common[v])} of {len(subset)} subset turns are answered by every system in the tables")

    # ---- aggregates per run and scope
    results = {v: {} for v in versions}
    for (s, v), tr in turns.items():
        maesc = turns.get(("maesc", v))
        cache = {}
        results[v][s] = {}
        for scope, uids in scope_sets(tr, subset, common[v], s in in_tables[v]).items():
            key = frozenset(uids)
            if key not in cache:
                cache[key] = aggregate([tr[u] for u in sorted(uids, key=uid_key)], maesc)
            results[v][s][scope] = cache[key]

    # ---- run summaries
    summary = {}
    for (s, v), info in runs.items():
        tr = turns[(s, v)]
        expected = set(samples[v]) if s in SINGLE_CALL else set(subset)
        summary[f"{s}/{v}"] = dict(
            system=s, version=v, records=len(tr), subset=len(set(tr) & subset),
            subset_coverage=100.0 * len(set(tr) & subset) / max(1, len(subset)),
            expected=len(expected), missing=len(expected - set(tr)),
            failed=sum(t["failed"] for t in tr.values()),
            none=sum(t["missing"] and not t["failed"] for t in tr.values()),
            stale=sum(t["stale"] for t in tr.values()),
            invalid_strategy=sum(t["invalid_strategy"] for t in tr.values()),
            failed_calls=call_total(tr, "n_failed_calls"), truncated_calls=call_total(tr, "n_truncated_calls"),
            paths=dict(Counter(t["path"] or "unknown" for t in tr.values())), in_tables=s in in_tables[v],
            **{k: info[k] for k in ("duplicates", "unreadable", "unknown_uid", "errors_unresolved")},
            meta=info.get("meta"), derived_from=info.get("derived_from"))
        sm = summary[f"{s}/{v}"]
        if sm["stale"]:
            warn(f"{s}/{v}: {sm['stale']} records differ from the current test data (reference/post); "
                 "was the data regenerated after the run?")

    # ---- gold reference register (target of the register metrics)
    reference_register = {}
    if prof is not None:
        for v in versions:
            rows = [M.register_match(prof, seeker_utterances(samples[v][u]), samples[v][u]["reference"])
                    for u in sorted(common[v], key=uid_key)]
            reference_register[v] = dict(M.register_summary(rows), n=len(rows))

    # ---- strategy stability
    stability = {v: {} for v in versions if v != "en"}
    if "en" in versions:
        for s in systems:
            if s in DERIVED:
                continue
            for v in stability:
                en_sys = s if (s, "en") in turns else EN_COUNTERPART.get(s)
                if (s, v) not in turns or en_sys is None or (en_sys, "en") not in turns:
                    continue
                A, B = turns[(en_sys, "en")], turns[(s, v)]
                if all(t["pred"] == "None" for t in list(A.values()) + list(B.values())):
                    continue  # never predicts a strategy
                entry = {}
                scopes = {"all": set(A) & set(B)}
                if s in in_tables[v] and en_sys in in_tables["en"]:
                    scopes["common"] = common["en"] & common[v]
                for scope, uids in scopes.items():
                    entry[scope] = dict(M.strategy_stability({u: A[u]["pred"] for u in uids},
                                                             {u: B[u]["pred"] for u in uids}), en_system=en_sys)
                stability[v][s] = entry

    # ---- significance: REF_SYSTEM against every other system of the version, common turns
    significance = {}
    for v in versions:
        if REF_SYSTEM not in in_tables[v]:
            continue
        significance[v] = {}
        uids = sorted(common[v], key=uid_key)
        for s in in_tables[v]:
            if s == REF_SYSTEM:
                continue
            entry = {}
            for key, _, _ in SIG_METRICS:
                pairs = [(turns[(REF_SYSTEM, v)][u][key], turns[(s, v)][u][key]) for u in uids]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if pairs:
                    entry[key] = M.paired_bootstrap([a for a, _ in pairs], [b for _, b in pairs], args.n_boot,
                                                    args.seed)
            significance[v][s] = entry

    # ---- outputs
    eval_dir = os.path.join(args.out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    meta = {
        "created": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "runs_dir": os.path.abspath(args.runs_dir), "systems": systems, "versions": versions,
        "subset_size": len(subset), "min_coverage": args.min_coverage,
        "common_turns": {v: len(common[v]) for v in versions}, "in_tables": in_tables,
        "tokenizer": f"NFC, lowercase, regex package: {M.TOKEN_PATTERN}",
        "bleu": "nltk corpus_bleu, cumulative uniform weights, one reference, SmoothingFunction().method3",
        "f1": "ParlAI unigram F1 (no punctuation, no a/an/the), mean over turns",
        "rouge_l": "LCS F-measure (beta=1) on word tokens, no stemming, mean over turns",
        "chrf_signature": M.chrf_signature(),
        "bertscore": {"model": bert.model_type, "hash": bert.hash} if bert else None,
        "profiler": args.profiler, "n_boot": args.n_boot, "seed": args.seed,
        "missing_responses": "scored as empty strings by quality metrics; skipped by register/length statistics",
    }
    with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(clean({"meta": meta, "runs": summary, "results": results, "reference_register": reference_register,
                         "stability": stability, "significance": significance}), f, ensure_ascii=False, indent=1)
    write_per_turn(os.path.join(eval_dir, "per_turn.jsonl"), turns)
    ctx = {"results": results, "common": common, "versions": versions, "in_tables": in_tables, "systems": systems,
           "reference_register": reference_register, "stability": stability, "significance": significance,
           "runs": summary, "turns": turns, "subset_size": len(subset), "n_boot": args.n_boot, "seed": args.seed,
           "profiler": PROFILER_NAMES[args.profiler]}
    write_tables(os.path.join(args.out_dir, "tables"), ctx)
    if not args.no_figures:
        write_figures(os.path.join(args.out_dir, "figures"), ctx)

    # ---- console summary
    say("runs (records / subset coverage / missing / failed / None / empty calls / truncated calls / stale):")
    for key, sm in summary.items():
        calls = " ".join(f"{'-' if sm[k] is None else sm[k]:>4}" for k in ("failed_calls", "truncated_calls"))
        say(f"  {key:<24} {sm['records']:>5} {sm['subset_coverage']:>6.1f}% {sm['missing']:>5} {sm['failed']:>4} "
            f"{sm['none']:>4} {calls} {sm['stale']:>4}{'' if sm['in_tables'] else '  (not in tables)'}")
    for v in versions:
        path = os.path.join(args.out_dir, "tables", f"main_{v}.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                print(f.read(), flush=True)
    say(f"wrote {eval_dir}, {os.path.join(args.out_dir, 'tables')}"
        + ("" if args.no_figures else f", {os.path.join(args.out_dir, 'figures')}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
