"""Talk to CodeMixESC (or any other system) in the terminal: type as a help-seeker, in
English or Hinglish, and the full multi-agent pipeline answers. --debug shows the seeker's
register profile, the path, the chosen strategy, the Register Gate and the LLM calls.

    python scripts/chat.py                       # CodeMixESC
    python scripts/chat.py --system maesc --debug
    python scripts/chat.py --dry_run             # stand-ins, no API key or models needed

Not a crisis service: the system is a research prototype and does not replace professional help.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc.esconv import json2natural  # noqa: E402
from codemixesc.systems import SYSTEMS, System  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="codemixesc", choices=[s for s in SYSTEMS if s != "pivot"] + ["pivot"])
    ap.add_argument("--model", default="gemma-4-26b-a4b-it")
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_system import build_components, default_gate
    spec = dict(SYSTEMS[args.system])
    tuned_delta, tuned_metric = default_gate()
    if spec.get("gate"):
        spec["gate_metric"] = tuned_metric
    llm, retriever, profiler = build_components(spec, args.model, args.dry_run)
    system = System(args.system, llm, retriever=retriever, profiler=profiler,
                    delta=args.delta if args.delta is not None else tuned_delta, **spec)
    history = []
    print(f"[{args.system}] Type your message (empty line or Ctrl-C to quit).")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        history.append({"role": "user", "content": text})
        count = len(history) + 1  # dialog index after this supporter turn, as in main.py
        sample = {"uid": f"chat-{len(history)}", "conv_id": -1, "turn": len(history), "strategy": "", "reference": "",
                  "context_msgs": list(history), "context": json2natural(history), "post": history[-1]["content"],
                  "early": count <= 5}
        rec = system.respond(sample, "chat")
        history.append({"role": "assistant", "content": rec["response"]})
        print(f"Supporter: {rec['response']}")
        if args.debug:
            g = rec.get("gate") or {}
            print(f"   [R={rec.get('R')}, path={rec['path']}, strategy={rec['pred_strategy']}, "
                  f"gate={'fired' if g.get('triggered') else 'no'}{' (rewrite kept)' if g.get('accepted') else ''}, "
                  f"calls={rec['n_calls']}, latency={rec['latency']:.1f}s]")


if __name__ == "__main__":
    main()
