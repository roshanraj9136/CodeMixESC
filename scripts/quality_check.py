"""Quality check of ESConv-HiEn (proposal, Section IV, step 3): a random ~20% of the test
conversations is rated for naturalness and meaning preservation on a 1-5 scale by the author
(and bilingual peers where available); conversations scoring below 3 are rewritten and
checked again. An LLM rater (gemma-4-31b-it) rates every conversation as a complement, and its
agreement with the human ratings is reported.

    python scripts/quality_check.py sample              # pick the 20% sample, write sheets + HTML views
    python scripts/quality_check.py llm                 # LLM ratings for all conversations
    (fill data/esconv_hien/qc/sheet_{level}.csv; several raters: copy the sheet, one file each)
    python scripts/quality_check.py report --sheets data/esconv_hien/qc/sheet_light.csv ...
    python scripts/quality_check.py fix                 # rewrite flagged conversations with feedback
    python scripts/quality_check.py llm --only_fixed    # check the rewritten conversations again
"""
import argparse
import csv
import glob
import html
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc.esconv import ROOT, hien_dir  # noqa: E402

LEVELS = ("light", "heavy")
JUDGE_MODEL = "gemma-4-31b-it"
THRESHOLD = 3

RATE_PROMPT = """You are a bilingual (Hindi-English) annotator checking a dataset. An English emotional support conversation was rewritten into Roman-script Hinglish (Hindi-English code-mixing as typed in chat by young Indians). Target mixing level: {level}.

Rate the Hinglish rewrite on two 1-5 scales:
- naturalness: 5 = reads like real chat Hinglish written by a native speaker; 3 = understandable but with awkward, literal or unnatural phrasing in several turns; 1 = unnatural, broken, or not Hinglish (e.g. plain English, Devanagari script).
- meaning: 5 = every turn keeps the meaning, emotion, facts and the supporter's strategy of the English original; 3 = some turns lose or change details; 1 = meaning, emotions or strategies are changed or missing.

Conversation (id, speaker, English original, Hinglish rewrite):
{turns}

Return only JSON: {{"naturalness": <1-5>, "meaning": <1-5>, "problem_turns": [<ids of turns with problems>], "comment": "<one or two sentences on the main problems>"}}"""

FEEDBACK = """

A reviewer rated an earlier rewrite of this conversation {nat}/5 for naturalness and {mean}/5 for meaning preservation. Their comment: "{comment}". Problem turns: {turns}. Avoid these problems: make every turn sound like natural chat Hinglish and keep the exact meaning, emotion and strategy of the English original."""


def qc_dir():
    path = os.path.join(hien_dir(), "qc")
    os.makedirs(path, exist_ok=True)
    return path


def load_level(level):
    with open(os.path.join(hien_dir(), f"test_{level}.json"), encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def conv_key(conv, i):
    return int(conv.get("esconv_index", i))


# ------------------------------------------------------------------ sample + sheets
def cmd_sample(args):
    n = len(load_level(LEVELS[0]))
    ids = sorted(random.Random(args.seed).sample(range(n), round(args.frac * n)))
    save_json({"conv_ids": ids, "frac": args.frac, "seed": args.seed}, os.path.join(qc_dir(), "sample.json"))
    for level in LEVELS:
        convs = load_level(level)
        with open(os.path.join(qc_dir(), f"sheet_{level}.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["level", "conv_id", "n_turns", "naturalness (1-5)", "meaning (1-5)", "comments"])
            for i in ids:
                w.writerow([level, i, len(convs[i]["dialog"]), "", "", ""])
        write_view(level, convs, ids, os.path.join(qc_dir(), f"view_{level}.html"))
    print(f"[qc] sample of {len(ids)} conversations: {ids}\n[qc] sheets and side-by-side views in {qc_dir()}")


def write_view(level, convs, ids, path):
    rows = [f"<h1>ESConv-HiEn {level}: conversations to rate</h1>",
            "<p>Rate each conversation for <b>naturalness</b> (does the Hinglish read like real chat?) and "
            "<b>meaning preservation</b> (same meaning, emotion and support strategy as the English), 1-5, "
            f"in sheet_{level}.csv.</p>"]
    for i in ids:
        rows.append(f"<h2>Conversation {i}</h2><table><tr><th>#</th><th>speaker</th><th>strategy</th>"
                    "<th>English original</th><th>Hinglish rewrite</th><th>CMI</th></tr>")
        for k, t in enumerate(convs[i]["dialog"]):
            strat = (t.get("annotation") or {}).get("strategy", "") if t["speaker"] == "supporter" else ""
            rows.append(f"<tr><td>{k}</td><td>{t['speaker']}</td><td>{html.escape(strat)}</td>"
                        f"<td>{html.escape(t.get('content_en', ''))}</td><td>{html.escape(t['content'])}</td>"
                        f"<td>{t.get('cmi', '')}</td></tr>")
        rows.append("</table>")
    style = ("<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;width:100%;"
             "margin-bottom:28px}td,th{border:1px solid #ccc;padding:4px 8px;vertical-align:top;text-align:left}"
             "th{background:#eee}</style>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><meta charset='utf-8'><title>ESConv-HiEn {level}</title>{style}" + "\n".join(rows))


# ------------------------------------------------------------------ LLM rater
def parse_rating(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        nat, mean = int(obj["naturalness"]), int(obj["meaning"])
    except (ValueError, KeyError, TypeError):
        return None
    if not (1 <= nat <= 5 and 1 <= mean <= 5):
        return None
    turns = [int(x) for x in obj.get("problem_turns", []) if str(x).lstrip("-").isdigit()]
    return {"naturalness": nat, "meaning": mean, "problem_turns": turns, "comment": str(obj.get("comment", ""))}


def rate_conv(llm, conv, level):
    turns = "\n".join(f"[{k}] {t['speaker']}: EN: {t.get('content_en', '').strip()} || HI: {t['content'].strip()}"
                      for k, t in enumerate(conv["dialog"]))
    prompt = RATE_PROMPT.format(level=level, turns=turns)
    for attempt in range(3):
        suffix = "" if attempt == 0 else f"\n\n(Attempt {attempt + 1}: return only the JSON object.)"
        r = parse_rating(llm(prompt + suffix, max_tokens=600, temperature=0.0, tag=f"hien-qc-{level}"))
        if r:
            return r
    return None


def cmd_llm(args):
    if args.dry_run:
        def llm(prompt, **kw):  # deterministic stand-in rater
            return '{"naturalness": %d, "meaning": 4, "problem_turns": [1], "comment": "stiff"}' % (2 + len(prompt) % 4)
    else:
        from codemixesc.llm import LLM
        llm = LLM(args.model, thinking="minimal")
    path = os.path.join(qc_dir(), "llm_ratings.json")
    ratings = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    fixed = json.load(open(os.path.join(qc_dir(), "fix_log.json"), encoding="utf-8")) if args.only_fixed else None
    import concurrent.futures as cf
    for level in LEVELS:
        convs = load_level(level)
        todo = range(len(convs))
        if args.sample_only:
            todo = json.load(open(os.path.join(qc_dir(), "sample.json")))["conv_ids"]
        if fixed is not None:
            todo = [int(k.split(":")[1]) for k in fixed if k.startswith(level + ":")]
        with cf.ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(lambda i: (i, rate_conv(llm, convs[i], level)), todo))
        for i, r in results:
            if r:
                key = f"{level}:{i}"
                r["round"] = ratings.get(key, {}).get("round", 0) + (1 if fixed is not None else 0)
                ratings[key] = r
        save_json(ratings, path)
        got = [ratings[f"{level}:{i}"] for i in todo if f"{level}:{i}" in ratings]
        if got:
            print(f"[qc] {level}: {len(got)} rated, mean naturalness {sum(r['naturalness'] for r in got) / len(got):.2f}, "
                  f"mean meaning {sum(r['meaning'] for r in got) / len(got):.2f}, "
                  f"{sum(min(r['naturalness'], r['meaning']) < THRESHOLD for r in got)} below {THRESHOLD}")


# ------------------------------------------------------------------ report
def read_sheets(paths):
    """{level:conv_id: {"naturalness": mean over raters, "meaning": ..., "n_raters": k}}"""
    acc = {}
    for p in paths:
        with open(p, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                nat, mean = row.get("naturalness (1-5)", "").strip(), row.get("meaning (1-5)", "").strip()
                if not nat or not mean:
                    continue
                key = f"{row['level'].strip()}:{int(row['conv_id'])}"
                acc.setdefault(key, []).append((float(nat), float(mean)))
    return {k: {"naturalness": sum(a for a, _ in v) / len(v), "meaning": sum(b for _, b in v) / len(v),
                "n_raters": len(v)} for k, v in acc.items()}


def weighted_kappa(a, b, k=5):
    """Quadratic-weighted Cohen's kappa for ratings in 1..k."""
    import numpy as np
    a, b = np.asarray(a, int) - 1, np.asarray(b, int) - 1
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[x, y] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    exp = np.outer(obs.sum(1), obs.sum(0)) / max(1, obs.sum())
    den = (w * exp).sum()
    return float(1 - (w * obs).sum() / den) if den else float("nan")


def cmd_report(args):
    from scipy.stats import spearmanr
    llm = json.load(open(os.path.join(qc_dir(), "llm_ratings.json"), encoding="utf-8"))
    sheets = args.sheets or sorted(glob.glob(os.path.join(qc_dir(), "sheet_*.csv")))
    human = read_sheets(sheets)
    flagged, rows = {}, []
    for level in LEVELS:
        L = {k: v for k, v in llm.items() if k.startswith(level + ":")}
        H = {k: v for k, v in human.items() if k.startswith(level + ":")}
        both = sorted(set(L) & set(H))
        row = {"level": level, "n_llm": len(L), "n_human": len(H)}
        for dim in ("naturalness", "meaning"):
            row[f"llm_{dim}"] = sum(v[dim] for v in L.values()) / len(L) if L else float("nan")
            row[f"human_{dim}"] = sum(v[dim] for v in H.values()) / len(H) if H else float("nan")
            hv, lv = [H[k][dim] for k in both], [L[k][dim] for k in both]
            if len(both) >= 3 and len(set(hv)) > 1 and len(set(lv)) > 1:  # correlations need variance
                row[f"spearman_{dim}"] = float(spearmanr(hv, lv).correlation)
                row[f"kappa_{dim}"] = weighted_kappa([round(x) for x in hv], lv)
        for k in set(L) | set(H):  # human ratings take precedence where they exist
            r = H.get(k) or L.get(k)
            if min(r["naturalness"], r["meaning"]) < THRESHOLD:
                flagged[k] = {"naturalness": r["naturalness"], "meaning": r["meaning"],
                              "source": "human" if k in H else "llm",
                              "comment": L.get(k, {}).get("comment", ""), "problem_turns": L.get(k, {}).get("problem_turns", [])}
        row["flagged"] = sum(1 for k in flagged if k.startswith(level + ":"))
        rows.append(row)
    save_json(flagged, os.path.join(qc_dir(), "flagged.json"))
    os.makedirs(os.path.join(ROOT, "results", "tables"), exist_ok=True)

    def f(x):
        return "–" if x != x else f"{x:.2f}"
    md = ["| Level | Human nat. | Human mean. | LLM nat. | LLM mean. | ρ nat. | ρ mean. | κw nat. | κw mean. | Flagged (<3) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lccccccccc}", r"\toprule",
           r"Level & \multicolumn{2}{c}{Human} & \multicolumn{2}{c}{LLM} & \multicolumn{2}{c}{Spearman $\rho$} & "
           r"\multicolumn{2}{c}{Weighted $\kappa$} & Flagged \\",
           r" & Nat. & Mean. & Nat. & Mean. & Nat. & Mean. & Nat. & Mean. & ($<$3) \\", r"\midrule"]
    for r in rows:
        vals = [r["human_naturalness"], r["human_meaning"], r["llm_naturalness"], r["llm_meaning"],
                r.get("spearman_naturalness", float("nan")), r.get("spearman_meaning", float("nan")),
                r.get("kappa_naturalness", float("nan")), r.get("kappa_meaning", float("nan"))]
        md.append(f"| {r['level']} (n={r['n_human']}/{r['n_llm']}) | " + " | ".join(f(v) for v in vals) + f" | {r['flagged']} |")
        tex.append(f"{r['level'].capitalize()} & " + " & ".join(f(v).replace("–", "--") for v in vals) + f" & {r['flagged']} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(ROOT, "results", "tables", "hien_quality.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(os.path.join(ROOT, "results", "tables", "hien_quality.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
    print("\n".join(md))
    print(f"[qc] {len(flagged)} conversations below {THRESHOLD} -> {os.path.join(qc_dir(), 'flagged.json')}")


# ------------------------------------------------------------------ fix
def cmd_fix(args):
    flagged = json.load(open(os.path.join(qc_dir(), "flagged.json"), encoding="utf-8"))
    if not flagged:
        print("[qc] nothing to fix")
        return
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from build_hien import Builder
    builder = Builder()
    log_path = os.path.join(qc_dir(), "fix_log.json")
    log = json.load(open(log_path, encoding="utf-8")) if os.path.exists(log_path) else {}
    for level in LEVELS:
        keys = [k for k in flagged if k.startswith(level + ":")]
        if not keys:
            continue
        path = os.path.join(hien_dir(), f"test_{level}.json")
        backup = os.path.join(hien_dir(), f"test_{level}.before_qc.json")
        if not os.path.exists(backup):
            shutil.copyfile(path, backup)
        convs = load_level(level)
        for k in keys:
            i = int(k.split(":")[1])
            fb = flagged[k]
            round_ = log.get(k, {}).get("rounds", 0) + 1
            extra = FEEDBACK.format(nat=round(fb["naturalness"]), mean=round(fb["meaning"]), comment=fb["comment"] or "-",
                                    turns=fb["problem_turns"] or "not given")
            extra += f" (Revision {round_}.)" if round_ > 1 else ""
            src = dict(convs[i])
            src["dialog"] = [dict(t, content=t.get("content_en", t["content"])) for t in convs[i]["dialog"]]
            for t in src["dialog"]:
                t.pop("content_en", None)
            new = builder.build_conv(src, level, extra=extra)
            new["esconv_index"] = convs[i].get("esconv_index", i)
            convs[i] = new
            log[k] = {"rounds": round_, "before": fb}
            print(f"[qc] rewrote {k} (round {round_})")
        save_json(convs, path)
    save_json(log, log_path)
    print("[qc] now re-rate the rewritten conversations: python scripts/quality_check.py llm --only_fixed "
          "(and check them manually)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("llm")
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--sample_only", action="store_true")
    p.add_argument("--only_fixed", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--sheets", nargs="*")
    sub.add_parser("fix")
    args = ap.parse_args()
    {"sample": cmd_sample, "llm": cmd_llm, "report": cmd_report, "fix": cmd_fix}[args.cmd](args)


if __name__ == "__main__":
    main()
