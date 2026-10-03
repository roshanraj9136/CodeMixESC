"""Selects case studies for the report from the run records, by fixed criteria (no cherry-picking
by hand), and writes them as Markdown and as LaTeX tables.

Categories (per Hinglish version; turns where both systems ran the full multi-agent pipeline
first, then the shortest contexts):
  rescue     MultiAgentESC's answer is (nearly) English while the seeker mixes, and CodeMixESC's
             answer is within the CMI gap the gate allows
  gate       the Register Gate fired and its rewrite was kept (pre-gate vs final response)
  strategy   CodeMixESC and MultiAgentESC chose different strategies and CodeMixESC's matches gold
  failure    CodeMixESC's final answer still misses the seeker's register (honest error analysis)

    python scripts/case_studies.py [--per_category 2] [--profiler hingbert|lexicon]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc.esconv import ROOT, all_samples, json2natural, seeker_utterances  # noqa: E402

LATEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
             "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(s):
    return "".join(LATEX_ESC.get(c, c) for c in str(s))


def load(runs_dir, system, version):
    path = os.path.join(runs_dir, system, f"{version}.jsonl")
    if not os.path.exists(path):
        return {}
    return {r["uid"]: r for r in map(json.loads, open(path, encoding="utf-8")) if r.get("response")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--per_category", type=int, default=2)
    ap.add_argument("--context_turns", type=int, default=4)
    ap.add_argument("--profiler", default="hingbert", choices=["hingbert", "lexicon"])
    args = ap.parse_args()
    if args.profiler == "lexicon":
        from codemixesc.testing import LexiconProfiler
        prof = LexiconProfiler()
    else:
        from codemixesc.profiler import Profiler
        prof = Profiler()

    cases = []
    for version in ("light", "heavy"):
        cmx, ma, pv = (load(args.runs_dir, s, version) for s in ("codemixesc", "maesc", "pivot"))
        if not cmx or not ma:
            continue
        samples = {s["uid"]: s for s in all_samples(version)}
        rows = []
        for uid in sorted(set(cmx) & set(ma), key=lambda u: tuple(map(int, u.split("-")))):
            s, c, m = samples[uid], cmx[uid], ma[uid]
            R = prof.profile(seeker_utterances(s))
            gap = {k: abs(prof.stats(r["response"])["cmi"] - R["cmi"]) for k, r in (("cmx", c), ("maesc", m))}
            ma_hi = prof.stats(m["response"])["hi_frac"]
            delta = (c.get("gate") or {}).get("delta", 0.2)
            gold = set(s["strategy"].split(" and "))
            cat = []
            if R["cmi"] >= 0.1 and ma_hi < 0.05 and gap["cmx"] <= delta:
                cat.append("rescue")
            if (c.get("gate") or {}).get("accepted"):
                cat.append("gate")
            if c["pred_strategy"] != m["pred_strategy"] and c["pred_strategy"] in gold:
                cat.append("strategy")
            if gap["cmx"] > delta:
                cat.append("failure")
            ctx = s["context_msgs"][-args.context_turns:]
            for k in cat:
                rows.append({"category": k, "version": version, "uid": uid, "R": R, "gap": gap, "gold": s["strategy"],
                             "full_pipeline": c.get("path") == "multi" and m.get("path") == "multi",
                             "context": ctx, "context_len": len(json2natural(ctx)), "reference": s["reference"],
                             "maesc": (m["pred_strategy"], m["response"]), "codemixesc": (c["pred_strategy"], c["response"]),
                             "pre_gate": c.get("pre_gate_response"), "pivot": (pv[uid]["pred_strategy"], pv[uid]["response"])
                             if uid in pv else None})
        for k in ("rescue", "gate", "strategy", "failure"):
            cases += sorted([r for r in rows if r["category"] == k],  # full multi-agent turns first, then short contexts
                            key=lambda r: (not r["full_pipeline"], r["context_len"]))[:args.per_category]

    md, tx = ["# Case studies (selected automatically, see scripts/case_studies.py)\n"], []
    for i, c in enumerate(cases, 1):
        md.append(f"## {i}. {c['category']} — {c['version']}, turn {c['uid']} (seeker CMI {c['R']['cmi']:.2f}, "
                  f"gold strategy: {c['gold']})")
        md += [f"> **{'Seeker' if m['role'] == 'user' else 'Supporter'}:** {m['content']}" for m in c["context"]]
        md.append("")
        md.append(f"- **Reference:** {c['reference']}")
        md.append(f"- **MultiAgentESC** [{c['maesc'][0]}] (CMI gap {c['gap']['maesc']:.2f}): {c['maesc'][1]}")
        if c["pivot"]:
            md.append(f"- **Translate-pivot** [{c['pivot'][0]}]: {c['pivot'][1]}")
        if c["category"] == "gate":
            md.append(f"- **CodeMixESC before the gate:** {c['pre_gate']}")
        md.append(f"- **CodeMixESC** [{c['codemixesc'][0]}] (CMI gap {c['gap']['cmx']:.2f}): {c['codemixesc'][1]}\n")
        rows = [(("Seeker" if m["role"] == "user" else "Supporter"), m["content"]) for m in c["context"]]
        rows += [("Reference", c["reference"]), (f"MultiAgentESC [{c['maesc'][0]}]", c["maesc"][1])]
        if c["pivot"]:
            rows.append((f"Pivot [{c['pivot'][0]}]", c["pivot"][1]))
        if c["category"] == "gate":
            rows.append(("CodeMixESC (pre-gate)", c["pre_gate"]))
        rows.append((f"CodeMixESC [{c['codemixesc'][0]}]", c["codemixesc"][1]))
        tx += [r"\begin{table}[t]", r"\centering\footnotesize",
               rf"\caption{{Case {i} ({tex(c['category'])}, {tex(c['version'])}, seeker CMI {c['R']['cmi']:.2f}).}}",
               rf"\label{{tab:case{i}}}", r"\begin{tabular}{@{}p{0.27\columnwidth}p{0.68\columnwidth}@{}}", r"\toprule"]
        tx += [rf"\textbf{{{tex(a)}}} & {tex(b)} \\" for a, b in rows]
        tx += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
    open(os.path.join(args.out_dir, "case_studies.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(os.path.join(args.out_dir, "tables", "case_studies.tex"), "w", encoding="utf-8").write("\n".join(tx) + "\n")
    print(f"[cases] {len(cases)} cases -> {os.path.join(args.out_dir, 'case_studies.md')}")


if __name__ == "__main__":
    main()
