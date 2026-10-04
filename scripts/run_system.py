"""Runs one system on one version of the test set and writes one JSON record per turn
(docs/RUN_FORMAT.md) to results/runs/{system}/{version}.jsonl.

The run is resumable: turns already in the output file are skipped, and turns that fail
(API errors after all retries) go to {version}.errors.jsonl and are retried on the next run.
Because every LLM call is cached, re-running or running an ablation that shares stages
with an earlier run costs nothing for the shared part.

Examples:
    python scripts/run_system.py --system codemixesc --version heavy
    python scripts/run_system.py --system zero_shot --version light          # all 1,210 turns
    python scripts/run_system.py --system maesc --version en --dry_run --limit 20
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc.esconv import ROOT, all_samples, dev_ids, dev_samples, sampled_uids  # noqa: E402
from codemixesc.systems import SYSTEMS, System, needs  # noqa: E402

DEFAULT_MODEL = "gemma-4-26b-a4b-it"
DELTA_PATH = os.path.join(ROOT, "results", "tuning", "delta.json")


def default_gate():
    """(delta, gate metric) from scripts/tune_delta.py; the proposal's initial 0.2 until it has run."""
    if os.path.exists(DELTA_PATH):
        tuned = json.load(open(DELTA_PATH, encoding="utf-8"))
        return float(tuned["delta"]), tuned.get("gate_metric", "hi_frac")
    return 0.2, "hi_frac"


def default_delta():
    return default_gate()[0]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def build_components(spec, model, dry_run, exclude_conv=()):
    """LLM, retriever and profiler for a system spec (stand-ins with dry_run)."""
    encoder, wants_profiler = needs(spec)
    if dry_run:
        from codemixesc.testing import FakeLLM, HashRetriever, LexiconProfiler
        return FakeLLM(), (HashRetriever(exclude_conv=exclude_conv) if encoder else None), \
            (LexiconProfiler() if wants_profiler else None)
    from codemixesc.llm import LLM
    llm, retriever, profiler = LLM(model, thinking="minimal"), None, None
    if encoder:
        from codemixesc.retriever import ENCODERS, Retriever
        if encoder == "mpnet-ft" and not os.path.exists(os.path.join(ENCODERS["mpnet-ft"], "config.json")):
            sys.exit("models/codemix-retriever is missing: run scripts/train_retriever.py first")
        retriever = Retriever(encoder, exclude_conv=exclude_conv)
    if wants_profiler:
        from codemixesc.profiler import Profiler
        profiler = Profiler()
    return llm, retriever, profiler


def load_done(path):
    """Records already written, newest per uid. A line cut off by a crash is skipped (the turn
    is simply run again)."""
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[rec["uid"]] = rec
    return done


def write_sorted(path, done):
    """Rewrites the records in turn order, atomically (resumed and parallel runs append out of
    order; a crash during the rewrite must not lose the file)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for uid in sorted(done, key=uid_key):
            f.write(json.dumps(done[uid], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


CONFIG_KEYS = ("base_system", "spec", "model", "delta", "split", "version")


def uid_key(uid):
    return tuple(int(x) for x in uid.split("-"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, help=", ".join(SYSTEMS))
    ap.add_argument("--version", required=True, choices=["en", "light", "heavy"])
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--subset", choices=["sampled", "all"], help="default: all for single-call systems, "
                                                                     "the fixed 200-turn sample for multi-agent ones")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--delta", type=float, help="Register Gate threshold (default: results/tuning/delta.json or 0.2)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N turns (smoke tests)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true", help="FakeLLM + stand-in profiler/retriever; writes to scratch/")
    # custom variants (e.g. extra ablations) on top of a named system
    ap.add_argument("--name", help="name of a custom variant (output folder)")
    ap.add_argument("--encoder", choices=["roberta", "labse", "mpnet", "mpnet-ft"])
    ap.add_argument("--no_register", action="store_true")
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--gate_metric", choices=["hi_frac", "cmi"], help="Register Gate distance (default hi_frac)")
    args = ap.parse_args()

    if args.system not in SYSTEMS:
        sys.exit(f"unknown system {args.system}; choose from {', '.join(SYSTEMS)}")
    spec = dict(SYSTEMS[args.system])
    if args.encoder:
        spec["encoder"] = args.encoder
    if args.no_register:
        spec["register"] = False
    if args.no_gate:
        spec["gate"] = False
    if args.gate_metric and args.gate_metric != "hi_frac":
        spec["gate_metric"] = args.gate_metric
    name = args.name or args.system
    if spec != SYSTEMS[args.system] and not args.name:
        sys.exit("a modified system needs --name so it does not overwrite the named system's results")
    tuned_delta, tuned_metric = default_gate()
    if spec.get("gate") and not args.gate_metric and tuned_metric != "hi_frac":
        spec["gate_metric"] = tuned_metric  # use the distance delta was tuned for
    if args.delta is not None and args.delta != tuned_delta and not args.name:
        sys.exit("a delta other than the tuned one makes a different system: give it a --name")
    if spec["kind"] == "pivot" and args.version == "en":
        sys.exit("the translate-pivot baseline is defined for the Hinglish versions only")

    if args.dry_run and args.version != "en" and not os.environ.get("CODEMIX_HIEN_DIR"):
        from codemixesc.testing import write_fake_hien
        os.environ["CODEMIX_HIEN_DIR"] = write_fake_hien(os.path.join(ROOT, "scratch", "fake_hien"))
    out_dir = args.out_dir or os.path.join(ROOT, "scratch" if args.dry_run else "results",
                                           "dry_runs" if args.dry_run else "runs")
    tag = args.version if args.split == "test" else f"dev_{args.version}"
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    out_path = os.path.join(out_dir, name, f"{tag}.jsonl")
    err_path = os.path.join(out_dir, name, f"{tag}.errors.jsonl")
    meta_path = os.path.join(out_dir, name, f"{tag}.meta.json")

    if args.split == "dev":
        samples, exclude = dev_samples(args.version), set(dev_ids())  # a dev query must not retrieve its own conversation
    else:
        samples, exclude = all_samples(args.version), ()
        subset = args.subset or ("all" if spec["kind"] in ("zero_shot", "fewshot_cot") else "sampled")
        if subset == "sampled":
            keep = set(sampled_uids(200))
            samples = [s for s in samples if s["uid"] in keep]
    if args.limit:
        samples = samples[:args.limit]
    delta = args.delta if args.delta is not None else tuned_delta
    meta = {"system": name, "base_system": args.system, "spec": spec, "version": args.version, "split": args.split,
            "model": "fake" if args.dry_run else args.model, "delta": delta if spec.get("gate") else None,
            "n_turns": len(samples), "limit": args.limit or None, "git_commit": git_commit(),
            "started": datetime.now(timezone.utc).isoformat()}

    done = load_done(out_path)
    if done and os.path.exists(meta_path):  # resuming: the earlier records must come from the same system
        prev = json.load(open(meta_path, encoding="utf-8"))
        diff = [k for k in CONFIG_KEYS if k in prev and prev[k] != meta[k]]
        if diff:
            sys.exit(f"{out_path} holds records of a different configuration ({', '.join(diff)} changed); "
                     "use --name for the new variant, or delete the old run")
    write_sorted(out_path, done)  # also drops a line cut off by a crash before appending
    # turns whose LLM calls failed (empty or blocked answers are never cached) are run again
    todo = [s for s in samples if s["uid"] not in done or done[s["uid"]].get("n_failed_calls")]
    retry = sum(1 for s in samples if s["uid"] in done and done[s["uid"]].get("n_failed_calls"))
    print(f"[run] {name} on {tag}: {len(samples)} turns, {len(done)} done ({retry} with failed calls, retried), "
          f"{len(todo)} to go (model {meta['model']}, encoder {spec.get('encoder')}, register "
          f"{spec.get('register', False)}, gate {spec.get('gate', False)}, delta {meta['delta']}, "
          f"gate metric {spec.get('gate_metric', 'hi_frac') if spec.get('gate') else None})", flush=True)

    llm, retriever, profiler = build_components(spec, args.model, args.dry_run, exclude)
    system = System(args.system, llm, retriever=retriever, profiler=profiler, delta=delta, **spec)
    lock = threading.Lock()
    t0, n_ok, n_err, calls, interrupted = time.time(), 0, 0, 0, False

    def job(sample):
        try:
            return sample, system.respond(sample, args.version), None
        except Exception as e:  # keep going; the turn is retried on the next run
            return sample, None, e

    ex = cf.ThreadPoolExecutor(max(1, args.workers))
    futures = [ex.submit(job, s) for s in todo]
    with open(out_path, "a", encoding="utf-8") as out:
        try:
            for fut in cf.as_completed(futures):
                sample, rec, e = fut.result()
                if e is not None:
                    with lock:
                        n_err += 1
                        with open(err_path, "a", encoding="utf-8") as err:
                            err.write(json.dumps({"uid": sample["uid"], "error": repr(e)[:1000],
                                                  "time": time.time()}) + "\n")
                    print(f"[run] turn {sample['uid']} failed: {e!r}"[:300], flush=True)
                    continue
                with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    n_ok += 1
                    calls += rec["n_calls"]
                    if n_ok % 10 == 0 or n_ok == len(todo):
                        el = time.time() - t0
                        eta = el / n_ok * (len(todo) - n_ok - n_err)
                        print(f"[run] {n_ok}/{len(todo)} done, {n_err} failed, {calls / n_ok:.1f} calls/turn, "
                              f"{el / 60:.1f} min elapsed, ETA {eta / 60:.1f} min", flush=True)
        except KeyboardInterrupt:  # stop queued turns from making API calls; keep what is done
            interrupted = True
            print("[run] interrupted: cancelling the remaining turns", flush=True)
            ex.shutdown(wait=False, cancel_futures=True)
    if not interrupted:
        ex.shutdown(wait=True)

    done = load_done(out_path)
    write_sorted(out_path, done)
    missing = sum(1 for s in samples if s["uid"] not in done)
    failed_calls = sum(1 for s in samples if s["uid"] in done and done[s["uid"]].get("n_failed_calls"))
    meta.update(finished=datetime.now(timezone.utc).isoformat(), n_done=len(done), n_missing=missing,
                n_with_failed_calls=failed_calls, errors_this_run=n_err, interrupted=interrupted)
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=1)
    print(f"[run] wrote {out_path}: {len(done)} records, {missing} missing, {failed_calls} with failed LLM calls "
          f"(re-run to retry them)", flush=True)
    if interrupted:
        os._exit(130)  # do not wait for the API calls still in flight
    sys.exit(1 if missing else (2 if failed_calls else 0))


if __name__ == "__main__":
    main()
