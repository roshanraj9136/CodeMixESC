"""Summary of the real LLM calls logged in cache/llm_calls.jsonl, per model and pipeline step:
calls, finish reasons, empty and truncated answers, input/output/thinking tokens, latency.

Run it after a short pilot (e.g. run_system.py --limit 10) before the long experiments: a step
whose answers are often truncated (MAX_TOKENS) or empty needs a larger token budget, and a
large "think" column means the model spends the output budget on reasoning tokens.

    python scripts/llm_usage.py [--since HOURS] [--model gemma-4-26b-a4b-it]
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=os.path.join(ROOT, "cache", "llm_calls.jsonl"))
    ap.add_argument("--since", type=float, default=0, help="only calls of the last N hours")
    ap.add_argument("--model")
    args = ap.parse_args()
    if not os.path.exists(args.log):
        sys.exit(f"no call log at {args.log}")
    t0 = time.time() - args.since * 3600 if args.since else 0
    groups = defaultdict(list)
    for line in open(args.log, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("t", 0) < t0 or (args.model and r.get("model") != args.model):
            continue
        groups[(r.get("model"), re.sub(r"\d+$", "", r.get("tag") or "-"))].append(r)
    if not groups:
        sys.exit("no calls match")

    def avg(rs, k):
        xs = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
        return sum(xs) / len(xs) if xs else float("nan")
    print(f"{'model':24s} {'step':16s} {'calls':>6s} {'trunc%':>7s} {'other%':>7s} {'in':>7s} {'out':>6s} "
          f"{'think':>6s} {'lat(s)':>7s}  finish reasons")
    for (model, tag), rs in sorted(groups.items()):
        fin = Counter(str(r.get("finish")) for r in rs)
        trunc = sum(v for k, v in fin.items() if k in ("MAX_TOKENS", "length")) / len(rs)
        other = sum(v for k, v in fin.items() if k not in ("STOP", "stop", "MAX_TOKENS", "length")) / len(rs)
        flag = "  <-- check token budget" if trunc > 0.02 else ("  <-- blocked/empty answers" if other > 0.02 else "")
        print(f"{model[:24]:24s} {tag[:16]:16s} {len(rs):6d} {100 * trunc:6.1f}% {100 * other:6.1f}% "
              f"{avg(rs, 'in'):7.0f} {avg(rs, 'out'):6.0f} {avg(rs, 'think'):6.0f} {avg(rs, 'latency'):7.2f}  "
              f"{dict(fin)}{flag}")


if __name__ == "__main__":
    main()
