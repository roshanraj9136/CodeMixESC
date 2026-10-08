"""ESConv loading, the base paper's test/case-bank split, and turn-level samples.

The turn extraction replicates main.py of MultiAgentESC exactly: the first 100
conversations of dataset/ESConv.json are the test set, the rest form the case bank,
two consecutive supporter utterances are merged into one sample, and samples with
count <= 5 are answered by the single zero-shot agent.
"""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESCONV_PATH = os.path.join(ROOT, "data", "esconv", "ESConv.json")
HIEN_DIR = os.path.join(ROOT, "data", "esconv_hien")
# Size of the fixed turn sample of the multi-agent systems (the proposal: 200). The sample is
# prefix-consistent: sampled_uids(100) is a subset of sampled_uids(200), so a run on a smaller
# sample can later be extended without discarding any finished turn.
SAMPLE_N = int(os.environ.get("CODEMIX_SAMPLE_N", "200"))
STRATEGIES = ["Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
              "Affirmation and Reassurance", "Providing Suggestions", "Information", "Others"]


def load_esconv():
    with open(ESCONV_PATH, encoding="utf-8") as f:
        return json.load(f)


def split(dataset):
    return dataset[:100], dataset[100:]


def hien_dir():
    """ESConv-HiEn directory; CODEMIX_HIEN_DIR overrides it (tests and dry runs)."""
    return os.environ.get("CODEMIX_HIEN_DIR") or HIEN_DIR


def load_version(version):
    """version: 'en', 'light' or 'heavy'. Returns the 100 test conversations."""
    if version == "en":
        return split(load_esconv())[0]
    path = os.path.join(hien_dir(), f"test_{version}.json")
    if not os.path.exists(path) and os.environ.get("CODEMIX_ALLOW_PARTIAL"):
        # build_hien.py writes test_{version}.partial.json while some conversations are still
        # missing (their entries hold an "error" instead of a dialog); runs on it cover the
        # finished conversations, and re-running after the build completes adds the rest
        partial = path.replace(".json", ".partial.json")
        if os.path.exists(partial):
            path = partial
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dev_ids():
    with open(os.path.join(hien_dir(), "dev_conv_ids.json"), encoding="utf-8") as f:
        return json.load(f)


def load_dev(version):
    """The development conversations ('en' = the English originals), each with its ESConv index."""
    if version == "en":
        data = load_esconv()
        return [dict(data[i], esconv_index=i) for i in dev_ids()]
    with open(os.path.join(hien_dir(), f"dev_{version}.json"), encoding="utf-8") as f:
        return json.load(f)


def dev_samples(version):
    """Turn samples of the dev split; conv_id is the ESConv index, so dev uids never collide
    with test uids (test conversations are 0-99, dev conversations are >= 100)."""
    out = []
    for conv in load_dev(version):
        out.extend(turn_samples(conv, conv["esconv_index"]))
    return out


def seeker_utterances(sample):
    return [m["content"] for m in sample["context_msgs"] if m["role"] == "user"]


def json2natural(history):
    out = ""
    for u in history:
        role = "Assistant" if u["role"] == "assistant" else "User"
        out += f"{role}: {u['content'].strip()} "
    return out.strip()


def turn_samples(conv, conv_id):
    """Replicates the sample loop of MultiAgentESC/main.py for one conversation."""
    dialog = conv["dialog"]
    count, history, samples = 0, [], []
    while count < len(dialog):
        if count != 0 and dialog[count]["speaker"] == "supporter":
            s = {"conv_id": conv_id, "turn": count}
            if (count < len(dialog) - 1 and dialog[count + 1]["speaker"] != "supporter") or count == len(dialog) - 1:
                s["strategy"] = dialog[count]["annotation"]["strategy"]
                s["reference"] = dialog[count]["content"].strip()
                s["context_msgs"] = list(history)
                s["context"] = json2natural(history)
                s["post"] = history[-1]["content"]
                history.append({"content": dialog[count]["content"].strip(), "role": "assistant"})
                count += 1
            else:
                s["strategy"] = f"{dialog[count]['annotation']['strategy']} and {dialog[count + 1]['annotation']['strategy']}"
                s["reference"] = dialog[count]["content"].strip() + " " + dialog[count + 1]["content"].strip()
                s["context_msgs"] = list(history)
                s["context"] = json2natural(history)
                s["post"] = history[-1]["content"]
                history.append({"content": dialog[count]["content"].strip(), "role": "assistant"})
                history.append({"content": dialog[count + 1]["content"].strip(), "role": "assistant"})
                count += 2
            s["early"] = count <= 5
            s["uid"] = f"{conv_id}-{s['turn']}"
            s["problem_type"] = conv.get("problem_type")
            samples.append(s)
        else:
            history.append({"content": dialog[count]["content"].strip(),
                            "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"})
            count += 1
    return samples


def all_samples(version):
    out = []
    for i, conv in enumerate(load_version(version)):
        if "dialog" in conv:  # a conversation a partial build has not finished yet
            out.extend(turn_samples(conv, i))
    return out


def sampled_uids(n=None, seed=42):
    """The fixed random subset of supporter turns used for the multi-agent systems (n defaults to
    SAMPLE_N). Identical uids exist in every version because the versions are parallel."""
    n = n or SAMPLE_N
    uids = [s["uid"] for s in all_samples("en")]
    rnd = random.Random(seed)
    return sorted(rnd.sample(uids, n), key=lambda u: tuple(int(x) for x in u.split("-")))


def case_bank(dataset=None, exclude_conv=()):
    """(post, response, strategy) triples of the case bank, as in get_quadruple() of the base code,
    with the conversation index and problem type kept for retrieval evaluation."""
    dataset = dataset or load_esconv()
    bank = []
    for ci, sample in enumerate(dataset[100:], start=100):
        if ci in exclude_conv:
            continue
        dialog = sample["dialog"]
        for c in range(len(dialog) - 1):
            if dialog[c]["speaker"] == "seeker" and dialog[c + 1]["speaker"] == "supporter":
                bank.append({"post": dialog[c]["content"].strip(), "response": dialog[c + 1]["content"].strip(),
                             "strategy": dialog[c + 1]["annotation"]["strategy"], "conv": ci,
                             "problem_type": sample["problem_type"]})
    return bank
