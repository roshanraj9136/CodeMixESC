"""Pairwise LLM judge and human-evaluation sheets: CodeMixESC against MultiAgentESC and the
Translate-Pivot (docs/EVALUATION.md).

Protocol. For each pair (a, b) and version, a seeded sample of turns on which both systems
answered is judged. One prompt per comparison shows the dialogue context and the two
responses as A and B and asks for a JSON verdict ("A", "B" or "tie") per dimension. Every
turn is judged in both orders (a shown as A, then a shown as B); per dimension the final
label is win (both orders prefer a), lose (both prefer b) or tie (anything else), which
cancels the judge's position bias. Identical responses are a tie without a call.
Dimensions: the five of ESConv's human evaluation (Liu et al., 2021), asked of single
responses (Fluency, Identification, Comforting, Suggestion, Overall), plus Language
Naturalness.

Human evaluation. --export_human writes one sheet for bilingual volunteers (same turns as the
LLM judge, A/B order randomised per item, system names hidden; the key goes to a separate
file). --import_human reads the filled copies (one CSV per volunteer), takes the majority label
per item and reports win/tie/lose, Cohen's kappa with the LLM judge and between volunteers.

Usage:
    python scripts/judge.py                                 # gemma-4-31b-it, light + heavy, 100 turns each
    python scripts/judge.py --versions en,light,heavy
    python scripts/judge.py --dry_run                       # fake judge, writes to scratch/judge_dry_run
    python scripts/judge.py --export_human --human_n 50
    python scripts/judge.py --import_human results/human/filled/*.csv
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc import metrics as M  # noqa: E402
from codemixesc.esconv import ROOT  # noqa: E402


def _load_evaluate():
    """scripts/evaluate.py under a private module name (run loading and table writing are
    shared; a pip package called `evaluate` must not shadow it)."""
    spec = importlib.util.spec_from_file_location(
        "codemixesc_evaluate_script", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load_evaluate()

JUDGE_MODEL = "gemma-4-31b-it"
DEFAULT_PAIRS = "codemixesc:maesc,codemixesc:pivot"
DIMENSIONS = [  # key, name, question (ESConv's human-evaluation questions, asked of single responses)
    ("fluency", "Fluency", "Which response is more fluent and understandable?"),
    ("identification", "Identification",
     "Which response explores the user's situation more in depth and is more helpful in identifying the "
     "user's problem?"),
    ("comforting", "Comforting", "Which response is more skillful in comforting the user?"),
    ("suggestion", "Suggestion", "Which response gives more helpful suggestions for the user's problem?"),
    ("overall", "Overall", "Generally, which response is the better emotional support, the one the user would prefer?"),
    ("naturalness", "Language Naturalness",
     "Whose language (script, Hindi-English mix, wording) sounds more natural for this particular user, "
     "given how the user writes?"),
]
KEYS = [k for k, _, _ in DIMENSIONS]
A_MARK, B_MARK, END_MARK = "### Response A\n", "\n\n### Response B\n", "\n\nCompare the two responses"

PROMPT = """You are an expert in emotional support conversations. Many users of this support chat write in Hinglish, a mix of Hindi and English that is usually typed in Roman script.

Below is a conversation between a User who is seeking emotional support and an Assistant, followed by two candidate responses for the Assistant's next turn.

### Dialogue context
{context}

""" + A_MARK + "{a}" + B_MARK + "{b}" + END_MARK + """ on each dimension:
{dimensions}

Judge each dimension on its own. Ignore the order in which the responses are shown, and do not prefer a response just because it is longer. Answer "tie" when the two responses are equally good on a dimension, or when the dimension applies to neither of them (for example, neither gives a suggestion).

Return only a JSON object that starts with a one-sentence comparison:
{{"analysis": "<one sentence>", {keys}}}
Every verdict must be "A", "B" or "tie"."""

_VERDICT = {"a": "A", "response a": "A", "b": "B", "response b": "B", "tie": "tie", "t": "tie", "equal": "tie",
            "same": "tie", "both": "tie", "neither": "tie", "draw": "tie", "=": "tie", "none": "tie"}
_KEY_ALIASES = {"fluency": "fluency", "identification": "identification", "comforting": "comforting",
                "suggestion": "suggestion", "suggestions": "suggestion", "overall": "overall",
                "naturalness": "naturalness", "language naturalness": "naturalness",
                "language_naturalness": "naturalness"}


def say(msg):
    print(f"[judge] {msg}", flush=True)


def warn(msg):
    print(f"[judge] WARNING: {msg}", flush=True)


# ================================================================== prompt and parsing
def format_context(sample):
    return "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'].strip()}"
                     for m in sample["context_msgs"])


def build_prompt(sample, response_a, response_b):
    dims = "\n".join(f"- {name} ({key}): {question}" for key, name, question in DIMENSIONS)
    keys = ", ".join(f'"{k}": "A/B/tie"' for k in KEYS)
    return PROMPT.format(context=format_context(sample), a=response_a.strip(), b=response_b.strip(),
                         dimensions=dims, keys=keys)


def verdict(value):
    """'A', 'B' or 'tie' from a judge's or volunteer's answer; None if unreadable."""
    return _VERDICT.get(re.sub(r"\s+", " ", str(value or "")).strip().strip("\"'.*[]()").lower())


def parse_verdicts(text):
    """{dimension: 'A' | 'B' | 'tie'} from a judge answer (JSON, possibly fenced or with stray
    text around it); None unless every dimension has a valid verdict."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    found = {}
    m = re.search(r"\{.*\}", text, re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict):
        items = obj.items()
    else:  # broken JSON: take "key": "verdict" pairs wherever they are
        items = re.findall(r'["\']?([A-Za-z][A-Za-z _]*?)["\']?\s*:\s*["\']?([A-Za-z=]+)["\']?', text)
    for k, v in items:
        key = _KEY_ALIASES.get(re.sub(r"\s+", " ", str(k)).strip().lower())
        if key and key not in found and isinstance(v, str) and verdict(v):
            found[key] = verdict(v)
    return {k: found[k] for k in KEYS} if all(k in found for k in KEYS) else None


def combine(v_ab, v_ba):
    """Final label per dimension from the two orders: v_ab has system a shown as A, v_ba has
    a shown as B. win = both orders prefer a, lose = both prefer b, tie = anything else."""
    out = {}
    for k in KEYS:
        first = {"A": "a", "B": "b", "tie": "tie"}[v_ab[k]]
        second = {"A": "b", "B": "a", "tie": "tie"}[v_ba[k]]
        out[k] = "win" if first == second == "a" else ("lose" if first == second == "b" else "tie")
    return out


def sample_uids(candidates, n, seed=42):
    """Seeded sample, stable under small changes of the candidate set: uids ranked by
    sha256(seed:uid) and the first n kept, so samples of different pairs and versions overlap
    as much as their candidates allow. Returned in ranking order."""
    ranked = sorted(candidates, key=lambda u: hashlib.sha256(f"{seed}:{u}".encode("utf-8")).hexdigest())
    return ranked[:n]


def answered(rec):
    return rec is not None and "error" not in rec and not M.is_missing(rec.get("response"))


def same_text(a, b):
    return re.sub(r"\s+", " ", a).strip() == re.sub(r"\s+", " ", b).strip()


# ================================================================== LLMs
class FakeJudgeLLM:
    """Deterministic stand-in for the judge (--dry_run and tests): prefers the longer response
    on fluency and overall, decides the other dimensions by a hash of the prompt, and answers
    the first attempt of every fourth prompt with prose, so the retry path runs."""

    model = "fake-judge"

    def __init__(self):
        self.n_calls = 0

    def chat(self, messages, system=None, temperature=0.0, max_tokens=400, tag="", json_mode=False):
        self.n_calls += 1
        prompt = messages[-1]["content"]
        h = int(hashlib.md5(prompt.encode("utf-8")).hexdigest(), 16)
        meta = {"cached": False, "latency": 0.0, "calls": 1}
        if h % 4 == 0 and "(Attempt" not in prompt:
            return "Both responses are supportive and kind.", meta
        a = prompt.split(A_MARK, 1)[1].split(B_MARK, 1)[0]
        b = prompt.split(B_MARK, 1)[1].split(END_MARK, 1)[0]
        v = {k: ["A", "B", "tie"][(h >> (2 * i)) % 3] for i, k in enumerate(KEYS)}
        v["fluency"] = v["overall"] = "A" if len(a) > len(b) else ("B" if len(b) > len(a) else "tie")
        return "```json\n" + json.dumps({"analysis": "dry run", **v}) + "\n```", meta


def judge_order(llm, prompt, attempts=3, max_tokens=600, json_mode=False):
    """(verdicts or None, raw answer, attempts used). The client caches by prompt, so a retry
    appends an attempt note; otherwise it would return the same cached unusable answer."""
    raw = ""
    for k in range(attempts):
        p = prompt if k == 0 else prompt + (f"\n\n(Attempt {k + 1}: reply with one JSON object with exactly the keys "
                                            f"analysis, {', '.join(KEYS)}; every verdict is \"A\", \"B\" or \"tie\".)")
        raw, _ = llm.chat([{"role": "user", "content": p}], temperature=0.0, max_tokens=max_tokens,
                          tag="pairwise_judge", json_mode=json_mode)
        v = parse_verdicts(raw)
        if v:
            return v, raw, k + 1
    return None, raw, attempts


# ================================================================== LLM judge
def judge_item(llm, sample, uid, version, a, b, ra, rb, args):
    item = {"uid": uid, "version": version, "a": a, "b": b, "response_a": ra, "response_b": rb,
            "identical": same_text(ra, rb), "model": llm.model}
    if item["identical"]:
        item.update(order_ab=None, order_ba=None, final={k: "tie" for k in KEYS})
        return item
    orders = {}
    for name, (x, y) in (("order_ab", (ra, rb)), ("order_ba", (rb, ra))):
        try:
            v, raw, n = judge_order(llm, build_prompt(sample, x, y), args.attempts, args.max_tokens, args.json_mode)
        except Exception as e:  # API failure after the client's retries: the item is reported as failed
            v, raw, n = None, f"error: {e!r}"[:500], 0
        orders[name] = {"verdicts": v, "attempts": n, "raw": raw}
    item.update(orders)
    ok = orders["order_ab"]["verdicts"] and orders["order_ba"]["verdicts"]
    item["final"] = combine(orders["order_ab"]["verdicts"], orders["order_ba"]["verdicts"]) if ok else None
    return item


def win_tie_lose(finals):
    """finals: one {dimension: win/tie/lose} per turn (a dimension may be None: not judged).
    Win/tie/lose % per dimension over the turns judged on it, and the two-sided sign test of
    wins against losses (ties left out)."""
    out = {"n": len(finals)}
    for k in KEYS:
        labels = [f[k] for f in finals if f.get(k)]
        c, n = Counter(labels), len(labels)
        out[k] = {"win": 100.0 * c["win"] / n if n else None, "tie": 100.0 * c["tie"] / n if n else None,
                  "lose": 100.0 * c["lose"] / n if n else None, "n": n, "n_win": c["win"], "n_tie": c["tie"],
                  "n_lose": c["lose"], "p_sign": M.sign_test(c["win"], c["lose"])}
    return out


def summarize(items):
    """win_tie_lose() over the judged items, plus the judge's position-bias diagnostics: how
    often both orders agree, and how often a decided verdict picks the first position."""
    done = [it for it in items if it.get("final")]
    out = win_tie_lose([it["final"] for it in done])
    out.update(n_failed=len(items) - len(done), n_identical=sum(it["identical"] for it in done))
    judged = [it for it in done if not it["identical"]]
    consistent, first, decided = 0, 0, 0
    for it in judged:
        for k in KEYS:
            x, y = it["order_ab"]["verdicts"][k], it["order_ba"]["verdicts"][k]
            consistent += {"A": "a", "B": "b", "tie": "tie"}[x] == {"A": "b", "B": "a", "tie": "tie"}[y]
            for v in (x, y):
                decided += v != "tie"
                first += v == "A"
    out["order_consistency"] = 100.0 * consistent / (len(judged) * len(KEYS)) if judged else None
    out["first_position_rate"] = 100.0 * first / decided if decided else None
    return out


def wtl_cell(s):
    if not s or s.get("win") is None:
        return None
    mark = "‡" if s["p_sign"] < 0.01 else ("†" if s["p_sign"] < 0.05 else "")
    tex_mark = r"$^{\ddagger}$" if s["p_sign"] < 0.01 else (r"$^{\dagger}$" if s["p_sign"] < 0.05 else "")
    text = f"{s['win']:.0f} / {s['tie']:.0f} / {s['lose']:.0f}"
    return {"md": text + mark, "tex": text + tex_mark,
            "csv": {"win": s["win"], "tie": s["tie"], "lose": s["lose"], "p_sign": s["p_sign"]}}


def write_wtl_table(tables_dir, name, rows_in, caption, extra_cols=()):
    """rows_in: list of (version, a, b, summary dict, extra values); one block per version."""
    rows = []
    order = {v: i for i, v in enumerate(ev.VERSIONS)}
    for version, a, b, s, extra in sorted(rows_in, key=lambda r: order.get(r[0], len(order))):
        row = {"version": ev.VNAMES.get(version, version), "comparison": f"{ev.name_of(a)} vs {ev.name_of(b)}",
               "n": s["n"], "_block": version, **extra}
        row.update({k: wtl_cell(s.get(k)) for k in KEYS})
        rows.append(row)
    cols = [ev.Col("version", "Version", fmt=None, merge=True), ev.Col("comparison", "Comparison", fmt=None),
            ev.Col("n", "n", fmt="d")] + [ev.Col(k, name) for k, name, _ in DIMENSIONS] + list(extra_cols)
    ev.write_table(tables_dir, name, cols, rows, caption,
                   "Cells: win / tie / lose in % of the turns; † p < 0.05, ‡ p < 0.01 (two-sided sign test of "
                   "wins against losses).")


def run_judge(args, samples):
    llm = FakeJudgeLLM() if args.dry_run else _real_llm(args)
    judge_dir = os.path.join(args.out_dir, "judge")
    os.makedirs(judge_dir, exist_ok=True)
    table_rows, summary = [], {}
    for a, b in args.pairs:
        for v in args.versions:
            runs = {}
            for s in (a, b):
                path = os.path.join(args.runs_dir, s, f"{v}.jsonl")
                if os.path.exists(path):
                    runs[s] = ev.load_run(path)[0]
            if len(runs) < 2 or v not in samples:
                say(f"{a} vs {b} on {v}: run or test data missing, skipped")
                continue
            cands = [u for u in runs[a] if u in runs[b] and u in samples[v] and answered(runs[a][u])
                     and answered(runs[b][u])]
            uids = sample_uids(cands, args.n, args.seed)
            if len(uids) < args.n:
                warn(f"{a} vs {b} on {v}: only {len(uids)} turns where both answered")
            say(f"{a} vs {b} on {v}: {len(uids)} turns ({llm.model})")
            with cf.ThreadPoolExecutor(max(1, args.workers)) as ex:
                futs = [ex.submit(judge_item, llm, samples[v][u], u, v, a, b, M.hypothesis(runs[a][u]["response"]),
                                  M.hypothesis(runs[b][u]["response"]), args) for u in uids]
                items = [f.result() for f in futs]
            items.sort(key=lambda it: ev.uid_key(it["uid"]))
            with open(os.path.join(judge_dir, f"{a}_vs_{b}_{v}.jsonl"), "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            s = summarize(items)
            summary[f"{a}_vs_{b}_{v}"] = s
            table_rows.append((v, a, b, s, {}))
            say(f"  overall win/tie/lose {s['overall']['win'] or 0:.0f}/{s['overall']['tie'] or 0:.0f}/"
                f"{s['overall']['lose'] or 0:.0f} %, {s['n_failed']} failed, order consistency "
                f"{s['order_consistency'] or 0:.0f} %, first position chosen in {s['first_position_rate'] or 0:.0f} % "
                f"of decided verdicts")
    if not table_rows:
        warn("nothing was judged")
        return 1
    meta = {"created": datetime.now(timezone.utc).isoformat(), "model": llm.model, "n": args.n, "seed": args.seed,
            "pairs": [f"{a}:{b}" for a, b in args.pairs], "versions": args.versions, "dry_run": args.dry_run}
    with open(os.path.join(judge_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(ev.clean({"meta": meta, "results": summary}), f, ensure_ascii=False, indent=1)
    write_wtl_table(os.path.join(args.out_dir, "tables"), "judge", table_rows,
                    f"Pairwise LLM judge ({llm.model}, both response orders) on a seeded sample of turns.")
    return 0


def _real_llm(args):
    from codemixesc.llm import LLM
    return LLM(args.model, thinking=None if args.thinking == "none" else args.thinking)


# ================================================================== human evaluation
HUMAN_INSTRUCTIONS = """Human evaluation of emotional support responses (CodeMixESC)

Each row of sheet.csv shows a conversation between a User who is seeking emotional support and an
Assistant, and two candidate responses (A and B) for the Assistant's next turn. The users often
write Hinglish (Hindi and English mixed, in Roman script).

For every dimension, write A if response A is better, B if response B is better, or tie if they
are equally good or the dimension applies to neither. Judge each dimension on its own; do not
prefer a response because it is longer. Please do not discuss items with other volunteers.

{dimensions}

Save the filled sheet as CSV (UTF-8) with your name in the file name, e.g. sheet_priya.csv.
"""


def export_human(args, samples):
    """One sheet for all volunteers: the first --human_n turns (in sampling order) of every
    pair and version, A/B randomised per item, items shuffled, system names only in the key."""
    human_dir = os.path.join(args.out_dir, "human")
    os.makedirs(human_dir, exist_ok=True)
    items = []
    for a, b in args.pairs:
        for v in args.versions:
            ra, rb = (os.path.join(args.runs_dir, s, f"{v}.jsonl") for s in (a, b))
            if not (os.path.exists(ra) and os.path.exists(rb)) or v not in samples:
                say(f"{a} vs {b} on {v}: run or test data missing, skipped")
                continue
            A, B = ev.load_run(ra)[0], ev.load_run(rb)[0]
            cands = [u for u in A if u in B and u in samples[v] and answered(A[u]) and answered(B[u])]
            ranked = [u for u in sample_uids(cands, args.n, args.seed)
                      if not same_text(M.hypothesis(A[u]["response"]), M.hypothesis(B[u]["response"]))]
            for u in ranked[:args.human_n]:
                flip = random.Random(f"{args.seed}:{a}:{b}:{v}:{u}").random() < 0.5
                shown = [(b, B[u]), (a, A[u])] if flip else [(a, A[u]), (b, B[u])]
                items.append({"pair": [a, b], "version": v, "uid": u, "A": shown[0][0], "B": shown[1][0],
                              "context": format_context(samples[v][u]),
                              "response_A": M.hypothesis(shown[0][1]["response"]),
                              "response_B": M.hypothesis(shown[1][1]["response"])})
    if not items:
        warn("no items to export")
        return 1
    random.Random(args.seed).shuffle(items)
    key = {}
    with open(os.path.join(human_dir, "sheet.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "context", "response_A", "response_B"] + KEYS + ["comment"])
        for i, it in enumerate(items, 1):
            item_id = f"H{i:03d}"
            w.writerow([item_id, it["context"], it["response_A"], it["response_B"]] + [""] * len(KEYS) + [""])
            key[item_id] = {k: it[k] for k in ("pair", "version", "uid", "A", "B")}
    with open(os.path.join(human_dir, "key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    dims = "\n".join(f"{key_}: {name}. {question}" for key_, name, question in DIMENSIONS)
    with open(os.path.join(human_dir, "instructions.txt"), "w", encoding="utf-8") as f:
        f.write(HUMAN_INSTRUCTIONS.format(dimensions=dims))
    say(f"wrote {len(items)} items to {os.path.join(human_dir, 'sheet.csv')} (key: key.json; keep it away from "
        "the volunteers)")
    return 0


def majority(labels):
    """The label most volunteers chose; 'tie' when no label has a strict majority."""
    c = Counter(labels).most_common()
    if not c:
        return None
    return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else "tie"


def import_human(args):
    with open(args.human_key or os.path.join(args.out_dir, "human", "key.json"), encoding="utf-8") as f:
        key = json.load(f)
    labels = defaultdict(lambda: defaultdict(dict))  # item -> annotator -> dimension -> win/tie/lose
    names = Counter()
    for path in args.import_human:
        annotator = os.path.splitext(os.path.basename(path))[0]
        names[annotator] += 1
        if names[annotator] > 1:  # two sheets with the same file name are still two volunteers
            annotator += f"_{names[annotator]}"
        n_filled = 0
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                item = (row.get("item") or "").strip()
                if item not in key:
                    continue
                k = key[item]
                for dim in KEYS:
                    v = verdict(row.get(dim))
                    if v is None:
                        continue
                    labels[item][annotator][dim] = "tie" if v == "tie" else ("win" if k[v] == k["pair"][0] else "lose")
                    n_filled += 1
        say(f"{annotator}: {n_filled} judgements")
    llm = {}  # (a, b, version, uid) -> final labels of the LLM judge
    for a, b, v in sorted({(*k["pair"], k["version"]) for k in key.values()}):
        path = os.path.join(args.out_dir, "judge", f"{a}_vs_{b}_{v}.jsonl")
        if not os.path.exists(path):
            warn(f"no LLM judge results for {a} vs {b} on {v}: no human-LLM agreement")
            continue
        for it in ev.read_jsonl(path)[0]:
            if it.get("final"):
                llm[(a, b, v, it["uid"])] = it["final"]
    groups = defaultdict(list)
    for item, k in key.items():
        if labels.get(item):
            groups[(k["version"], *k["pair"])].append(item)
    table_rows, summary = [], {}
    for (v, a, b), items in sorted(groups.items(), key=lambda g: (ev.VERSIONS.index(g[0][0]) if g[0][0] in ev.VERSIONS
                                                                   else 9, g[0][1:])):
        maj = {it: {d: majority([labels[it][ann][d] for ann in labels[it] if d in labels[it][ann]]) for d in KEYS}
               for it in items}
        s = win_tie_lose([{d: maj[it][d] for d in KEYS} for it in items])
        # agreement with the LLM judge (majority label vs the judge's final label)
        pairs = {d: [(maj[it][d], llm[(a, b, v, key[it]["uid"])][d]) for it in items
                     if maj[it][d] and (a, b, v, key[it]["uid"]) in llm] for d in KEYS}
        s["kappa_llm"] = {d: M.cohen_kappa([x for x, _ in p], [y for _, y in p]) for d, p in pairs.items()}
        pooled = [xy for p in pairs.values() for xy in p]
        s["kappa_llm_pooled"] = M.cohen_kappa([x for x, _ in pooled], [y for _, y in pooled])
        s["n_llm_overlap"] = len(pooled)
        # agreement between volunteers: mean pairwise Cohen's kappa over the shared judgements
        anns = sorted({ann for it in items for ann in labels[it]})
        kappas = []
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                pa = [(labels[it][anns[i]][d], labels[it][anns[j]][d]) for it in items for d in KEYS
                      if d in labels[it].get(anns[i], {}) and d in labels[it].get(anns[j], {})]
                kp = M.cohen_kappa([x for x, _ in pa], [y for _, y in pa])
                if kp is not None:
                    kappas.append(kp)
        s["annotators"] = anns
        s["kappa_annotators"] = M.mean(kappas)
        summary[f"{a}_vs_{b}_{v}"] = s
        table_rows.append((v, a, b, s, {"kappa_llm": s["kappa_llm_pooled"], "kappa_iaa": s["kappa_annotators"]}))
    if not table_rows:
        warn("no filled judgements found")
        return 1
    with open(os.path.join(args.out_dir, "human", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(ev.clean({"sheets": args.import_human, "results": summary}), f, ensure_ascii=False, indent=1)
    write_wtl_table(os.path.join(args.out_dir, "tables"), "human", table_rows,
                    "Human evaluation by bilingual volunteers (majority label per turn), with Cohen's kappa against "
                    "the LLM judge and between volunteers (pooled over dimensions).",
                    [ev.Col("kappa_llm", "κ LLM", "$\\kappa_{\\mathrm{LLM}}$"),
                     ev.Col("kappa_iaa", "κ volunteers", "$\\kappa_{\\mathrm{IAA}}$")])
    return 0


# ================================================================== main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=DEFAULT_PAIRS, help="comma-separated a:b; labels are from a's side")
    ap.add_argument("--versions", default="light,heavy")
    ap.add_argument("--n", type=int, default=100, help="turns per pair and version")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--thinking", default="minimal", help="thinking level of the judge ('none' to omit)")
    ap.add_argument("--json_mode", action="store_true", help="ask the API for JSON output (if the model supports it)")
    ap.add_argument("--max_tokens", type=int, default=600)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs_dir", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--out_dir", help="writes judge/, human/ and tables/ below it (default results/, "
                                      "or scratch/judge_dry_run with --dry_run)")
    ap.add_argument("--dry_run", action="store_true", help="fake judge LLM, no API calls")
    ap.add_argument("--export_human", action="store_true")
    ap.add_argument("--human_n", type=int, default=50, help="items per pair and version in the human sheet")
    ap.add_argument("--import_human", nargs="+", metavar="CSV", help="filled sheets, one per volunteer")
    ap.add_argument("--human_key", help="default: <out_dir>/human/key.json")
    ap.add_argument("--hien_dir", help="ESConv-HiEn directory (default: $CODEMIX_HIEN_DIR or data/esconv_hien)")
    ap.add_argument("--samples_file", help="JSON {version: [sample, ...]} instead of the test data (tests)")
    args = ap.parse_args(argv)
    args.pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p]
    if any(len(p) != 2 for p in args.pairs):
        ap.error("--pairs takes a:b items")
    args.versions = [v for v in args.versions.split(",") if v]
    args.out_dir = args.out_dir or (os.path.join(ROOT, "scratch", "judge_dry_run") if args.dry_run
                                    else os.path.join(ROOT, "results"))
    return args


def main(argv=None):
    args = parse_args(argv)
    ev.safe_console()
    if args.hien_dir:
        os.environ["CODEMIX_HIEN_DIR"] = os.path.abspath(args.hien_dir)
    if args.import_human:
        return import_human(args)
    samples = ev.load_samples(args.versions, args.samples_file)
    if args.export_human:
        return export_human(args, samples)
    return run_judge(args, samples)


if __name__ == "__main__":
    sys.exit(main())
