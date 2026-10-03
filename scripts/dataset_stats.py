"""Statistics of ESConv-HiEn for the report (dataset table and CMI histogram).

The CMI of every utterance is recomputed with the profiler as committed (not taken from the
values stored at build time). The English originals are profiled too: the share of their
words that HingBERT-LID tags as Hindi is the profiler's false-positive rate, which decides how
"plain English" (profiler.PLAIN_ENGLISH_CMI) must be set.

    python scripts/dataset_stats.py [--split test|dev] [--profiler hingbert|lexicon]
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc.esconv import ROOT, all_samples, dev_samples, load_dev, load_version, seeker_utterances  # noqa: E402
from codemixesc.profiler import PLAIN_ENGLISH_CMI, script_of  # noqa: E402

BANDS = {"light": (0.10, 0.30), "heavy": (0.30, 0.50)}  # as in scripts/build_hien.py
MIN_WORDS_FOR_BAND = 5


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def level_stats(convs, level, prof, samples):
    utts = [t for c in convs for t in c["dialog"]]
    texts = [t["content"].strip() for t in utts]
    st = prof.stats_many(texts)
    by = {"seeker": [], "supporter": []}
    for t, s in zip(utts, st):
        by[t["speaker"]].append(s)
    checked = [s for s in st if s["n_lang"] >= MIN_WORDS_FOR_BAND]
    out = {"conversations": len(convs), "utterances": len(utts), "words_per_utt": mean(len(x.split()) for x in texts),
           "cmi": mean(s["cmi"] for s in st), "cmi_seeker": mean(s["cmi"] for s in by["seeker"]),
           "cmi_supporter": mean(s["cmi"] for s in by["supporter"]), "hi_frac": mean(s["hi_frac"] for s in st),
           "scripts": dict(Counter(script_of(x) for x in texts))}
    if level in BANDS:
        lo, hi = BANDS[level]
        out["in_band"] = mean(lo - 1e-9 <= s["cmi"] <= hi + 1e-9 for s in checked)
        out["band_checked"] = len(checked)
        sims = [t["labse_sim"] for t in utts if "labse_sim" in t]
        out["labse_sim"] = mean(sims)
        out["regenerated"] = mean(t.get("regenerated", 0) > 0 for t in utts)
        out["words_per_utt_en"] = mean(len(t.get("content_en", "").split()) for t in utts)
    else:  # English originals: every word tagged HI is a false positive of the LID model
        n_hi, n_lang = sum(s["hi"] for s in st), sum(s["n_lang"] for s in st)
        out["false_hindi_rate"] = n_hi / max(1, n_lang)
    # what the agents see: the pooled seeker profile R at every turn
    Rs = [prof.profile(seeker_utterances(s)) for s in samples]
    out["turn_cmi_s"] = mean(R["cmi"] for R in Rs)
    out["turn_plain_english"] = mean(R["cmi"] < PLAIN_ENGLISH_CMI and R["dominant"] == "English" for R in Rs)
    out["_utt_cmi"] = [s["cmi"] for s in checked]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--profiler", default="hingbert", choices=["hingbert", "lexicon"])
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    if args.profiler == "lexicon":
        from codemixesc.testing import LexiconProfiler
        prof = LexiconProfiler()
    else:
        from codemixesc.profiler import Profiler
        prof = Profiler()
    stats = {}
    for level in ("en", "light", "heavy"):
        try:
            convs = load_version(level) if args.split == "test" else load_dev(level)
            samples = all_samples(level) if args.split == "test" else dev_samples(level)
        except FileNotFoundError:
            print(f"[stats] {args.split} {level}: not built yet, skipped")
            continue
        stats[level] = level_stats(convs, level, prof, samples)

    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "eval"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    json.dump({k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in stats.items()},
              open(os.path.join(args.out_dir, "eval", f"hien_stats_{args.split}.json"), "w"), indent=1)

    def g(v, key, fmt="{:.3f}", scale=1):
        x = v.get(key)
        return "–" if x is None or x != x else fmt.format(x * scale)
    rows = [("Conversations", "conversations", "{:.0f}", 1), ("Utterances", "utterances", "{:.0f}", 1),
            ("Words / utterance", "words_per_utt", "{:.1f}", 1), ("Mean CMI (all)", "cmi", "{:.3f}", 1),
            ("Mean CMI (seeker)", "cmi_seeker", "{:.3f}", 1), ("Mean CMI (supporter)", "cmi_supporter", "{:.3f}", 1),
            ("Hindi words (%)", "hi_frac", "{:.1f}", 100), ("In target band (%)", "in_band", "{:.1f}", 100),
            ("LaBSE sim. to English", "labse_sim", "{:.3f}", 1), ("Regenerated (%)", "regenerated", "{:.1f}", 100),
            ("Seeker CMI$_s$ per turn", "turn_cmi_s", "{:.3f}", 1),
            ("LID false-Hindi rate (%)", "false_hindi_rate", "{:.2f}", 100)]
    levels = list(stats)
    md = ["| | " + " | ".join(levels) + " |", "|---|" + "---|" * len(levels)]
    tex = [r"\begin{tabular}{l" + "c" * len(levels) + "}", r"\toprule",
           " & " + " & ".join({"en": "English", "light": "Light", "heavy": "Heavy"}[x] for x in levels) + r" \\",
           r"\midrule"]
    for label, key, fmt, scale in rows:
        cells = [g(stats[lv], key, fmt, scale) for lv in levels]
        md.append(f"| {label.replace('$_s$', '_s')} | " + " | ".join(cells) + " |")
        tex.append(label + " & " + " & ".join(c.replace("–", "--") for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(args.out_dir, "tables", f"hien_stats_{args.split}.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(os.path.join(args.out_dir, "tables", f"hien_stats_{args.split}.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    for lv in levels:
        ax.hist(stats[lv]["_utt_cmi"], bins=25, range=(0, 0.5), alpha=0.55, label=lv)
    for lo, hi in BANDS.values():
        ax.axvline(lo, color="grey", lw=0.6, ls="--")
        ax.axvline(hi, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("utterance CMI")
    ax.set_ylabel("utterances")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "figures", f"hien_cmi_hist_{args.split}.png"), dpi=200)
    print(f"[stats] tables in {os.path.join(args.out_dir, 'tables')}")


if __name__ == "__main__":
    main()
