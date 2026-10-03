"""Runs the whole experimental plan, one run after another (they share one API quota), most
important comparisons first so that a quota cut leaves the key results complete. Finished runs
are skipped; each run's output goes to logs/{system}_{version}.log.

    python scripts/run_all.py                    # everything, then evaluation
    python scripts/run_all.py --only maesc,codemixesc --versions light,heavy
    python scripts/run_all.py --dry_run          # whole plan with stand-ins (minutes)

On Windows, start it detached so it survives closing the terminal, e.g.
    Start-Process -WindowStyle Hidden -FilePath .venv\\Scripts\\python.exe -ArgumentList "-X utf8 scripts\\run_all.py" -RedirectStandardOutput logs\\run_all.out -RedirectStandardError logs\\run_all.err
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLAN = [  # (system, version) in priority order
    ("maesc", "light"), ("codemixesc", "light"), ("maesc", "heavy"), ("codemixesc", "heavy"),
    ("pivot", "light"), ("pivot", "heavy"), ("maesc", "en"), ("codemixesc", "en"),
    ("cmx_noft", "light"), ("cmx_noft", "heavy"), ("cmx_noxl", "light"), ("cmx_noxl", "heavy"),
    ("zero_shot", "en"), ("zero_shot", "light"), ("zero_shot", "heavy"),
    ("fewshot_cot", "en"), ("fewshot_cot", "light"), ("fewshot_cot", "heavy"),
]


def finished(out_dir, system, version):
    path = os.path.join(out_dir, system, f"{version}.meta.json")
    if not os.path.exists(path):
        return False
    meta = json.load(open(path, encoding="utf-8"))
    return meta.get("n_missing") == 0 and meta.get("n_done", 0) >= meta.get("n_turns", 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated systems")
    ap.add_argument("--versions", default="en,light,heavy")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_eval", action="store_true")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    versions = set(args.versions.split(","))
    out_dir = os.path.join(ROOT, "scratch", "dry_runs") if args.dry_run else os.path.join(ROOT, "results", "runs")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    failed = []
    for system, version in PLAN:
        if (only and system not in only) or version not in versions:
            continue
        if finished(out_dir, system, version):
            print(f"[all] {system}/{version}: done, skipped", flush=True)
            continue
        cmd = [sys.executable, "-X", "utf8", os.path.join(ROOT, "scripts", "run_system.py"), "--system", system,
               "--version", version, "--workers", str(args.workers)] + (["--dry_run"] if args.dry_run else [])
        log = os.path.join(ROOT, "logs", f"{system}_{version}{'_dry' if args.dry_run else ''}.log")
        t0 = time.time()
        print(f"[all] {system}/{version}: running (log {log})", flush=True)
        with open(log, "a", encoding="utf-8") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        print(f"[all] {system}/{version}: exit {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
        if code != 0:
            failed.append(f"{system}/{version}")
    if failed:
        print(f"[all] incomplete runs (re-run to resume): {', '.join(failed)}", flush=True)
    if not args.no_eval and not failed:
        cmd = [sys.executable, "-X", "utf8", os.path.join(ROOT, "scripts", "evaluate.py")]
        if args.dry_run:
            cmd += ["--runs_dir", out_dir, "--out_dir", os.path.join(ROOT, "scratch", "dry_eval"),
                    "--profiler", "lexicon", "--no_bertscore"]
        if os.path.exists(cmd[3]):
            print("[all] evaluating", flush=True)
            subprocess.call(cmd, cwd=ROOT)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
