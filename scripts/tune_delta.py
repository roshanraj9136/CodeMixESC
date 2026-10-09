"""Tunes the Register Gate threshold delta on the development set (proposal: initially 0.2,
tuned on a dev set).

CodeMixESC is run on dev turns without the gate. The gate is then applied once per turn
with the smallest delta of the grid. Whether a rewrite is accepted does not depend on delta
(it is kept only if its register is closer to the seeker's), so every delta of the grid can
be scored exactly without further LLM calls: for delta d, a turn is gated iff its register
distance (profiler.register_distance) exceeds d or the script differs.

Selection rule (cheapest threshold that gets essentially the best register match without
hurting quality): among the deltas whose pooled mean distance is within --tol of the best one
and whose chrF is at least the ungated chrF minus --max_chrf_drop, take the largest.

    python scripts/tune_delta.py                 # light + heavy dev conversations
    python scripts/tune_delta.py --dry_run
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc import agents as A  # noqa: E402
from codemixesc.esconv import ROOT, dev_ids, dev_samples  # noqa: E402
from codemixesc.systems import SYSTEMS, System  # noqa: E402

GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


def chrf(hyps, refs):
    from sacrebleu.metrics import CHRF
    return CHRF().corpus_score(hyps, [refs]).score


def run_level(level, args, llm, retriever, profiler):
    samples = dev_samples(level)
    if args.max_turns and len(samples) > args.max_turns:
        samples = sorted(random.Random(42).sample(samples, args.max_turns), key=lambda s: (s["conv_id"], s["turn"]))
    system = System("codemixesc", llm, retriever=retriever, profiler=profiler, gate=False,
                    encoder=args.encoder)
    lo = min(args.grid)

    def job(sample):
        rec = system.respond(sample, level)
        log = A.CallLog(llm)
        _, info = A.register_gate(log, profiler, sample["context"], rec["R"], rec["pred_strategy"],
                                  rec["pre_gate_response"], lo - 1e-9, args.gate_metric)
        cand = info.get("candidate") if info["accepted"] else None
        return {"uid": sample["uid"], "reference": sample["reference"], "R": rec["R"], "pre": rec["pre_gate_response"],
                "bad_script": bool(A.script_mismatch(rec["R"], {"script": info["script_before"]})), "gated": cand,
                "dist_pre": info["distance_before"], "dist_gated": info["distance_after"] if cand else None,
                "cmi_pre": abs(info["cmi_before"] - info["cmi_target"]),
                "cmi_gated": abs(info["cmi_after"] - info["cmi_target"]) if cand else None}

    with cf.ThreadPoolExecutor(args.workers) as ex:
        return list(ex.map(job, samples))


def score(rows, d):
    """Outcome of delta d (None = no gate): (final responses, per-turn gate distances, per-turn
    CMI gaps, number of gate calls)."""
    finals, dists, gaps, calls = [], [], [], 0
    for r in rows:
        fire = d is not None and (r["dist_pre"] > d or r["bad_script"])
        calls += fire
        use = fire and r["gated"] is not None
        finals.append(r["gated"] if use else r["pre"])
        dists.append(r["dist_gated"] if use else r["dist_pre"])
        gaps.append(r["cmi_gated"] if use else r["cmi_pre"])
    return finals, dists, gaps, calls


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="light,heavy")
    ap.add_argument("--grid", default=",".join(str(g) for g in GRID))
    ap.add_argument("--max_turns", type=int, default=80, help="dev turns per level (seeded sample; 0 = all)")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--max_chrf_drop", type=float, default=1.0)
    ap.add_argument("--encoder", default=SYSTEMS["codemixesc"]["encoder"])
    ap.add_argument("--gate_metric", default="hi_frac", choices=["hi_frac", "cmi"],
                    help="hi_frac: Hindi-share gap (CMI gap + dominant language); cmi: the proposal's CMI gap")
    ap.add_argument("--model", default="gemma-4-26b-a4b-it")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.grid = sorted(float(x) for x in args.grid.split(","))
    levels = args.levels.split(",")

    if args.dry_run:
        from codemixesc.testing import FakeLLM, HashRetriever, LexiconProfiler, write_fake_hien
        if not os.environ.get("CODEMIX_HIEN_DIR"):
            os.environ["CODEMIX_HIEN_DIR"] = write_fake_hien(os.path.join(ROOT, "scratch", "fake_hien"))
        llm, retriever, profiler = FakeLLM(), HashRetriever(exclude_conv=set(dev_ids())), LexiconProfiler()
    else:
        from codemixesc.llm import LLM
        from codemixesc.profiler import Profiler
        from codemixesc.retriever import ENCODERS, Retriever
        if args.encoder == "mpnet-ft" and not os.path.exists(os.path.join(ENCODERS["mpnet-ft"], "config.json")):
            sys.exit("models/codemix-retriever is missing: train the retriever first (or pass --encoder mpnet)")
        llm, profiler = LLM(args.model, thinking="minimal"), Profiler()
        retriever = Retriever(args.encoder, exclude_conv=set(dev_ids()))  # never retrieve the query's own conversation

    rows = {lv: run_level(lv, args, llm, retriever, profiler) for lv in levels}
    pooled = [r for lv in levels for r in rows[lv]]
    table = []
    for d in [None] + args.grid:  # None = no gate
        entry = {"delta": d}
        for name, rr in [(lv, rows[lv]) for lv in levels] + [("all", pooled)]:
            finals, dists, gaps, calls = score(rr, d)
            entry[name] = {"n": len(rr), "distance": sum(dists) / len(dists), "cmi_gap": sum(gaps) / len(gaps),
                           "gate_rate": calls / len(rr), "chrf": chrf(finals, [r["reference"] for r in rr])}
        table.append(entry)
    base = table[0]["all"]
    best = min(e["all"]["distance"] for e in table[1:])
    ok = [e for e in table[1:] if e["all"]["distance"] <= best + args.tol
          and e["all"]["chrf"] >= base["chrf"] - args.max_chrf_drop]
    # if every delta costs chrF, take the least harmful one (ties: the larger delta, fewer calls)
    chosen = max(ok, key=lambda e: e["delta"]) if ok else max(table[1:], key=lambda e: (e["all"]["chrf"], e["delta"]))

    out_dir = args.out or os.path.join(ROOT, "scratch" if args.dry_run else "results", "tuning")
    os.makedirs(out_dir, exist_ok=True)
    result = {"delta": chosen["delta"], "gate_metric": args.gate_metric,
              "rule": f"largest delta with pooled gate distance ({args.gate_metric}) <= best + {args.tol} "
                      f"and chrF >= ungated - {args.max_chrf_drop}",
              "levels": levels, "encoder": args.encoder, "model": "fake" if args.dry_run else args.model,
              "table": table}
    json.dump(result, open(os.path.join(out_dir, "delta.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(out_dir, "turns.jsonl"), "w", encoding="utf-8") as f:
        for lv in levels:
            for r in rows[lv]:
                f.write(json.dumps(dict(r, level=lv), ensure_ascii=False) + "\n")
    lines = ["| δ | " + " | ".join(f"{n} distance | {n} CMI gap | {n} gate rate | {n} chrF" for n in levels + ["all"])
             + " |", "|---|" + "---|---|---|---|" * (len(levels) + 1)]
    for e in table:
        cells = [f"{e[n]['distance']:.3f} | {e[n]['cmi_gap']:.3f} | {100 * e[n]['gate_rate']:.0f}% | {e[n]['chrf']:.1f}"
                 for n in levels + ["all"]]
        mark = " **(chosen)**" if e is chosen else ""
        lines.append(f"| {'no gate' if e['delta'] is None else e['delta']}{mark} | " + " | ".join(cells) + " |")
    open(os.path.join(out_dir, "delta.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[tune] chosen delta = {chosen['delta']} -> {os.path.join(out_dir, 'delta.json')}")


if __name__ == "__main__":
    main()
