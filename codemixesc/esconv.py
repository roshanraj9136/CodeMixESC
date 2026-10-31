import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESCONV_PATH = os.path.join(ROOT, "data", "esconv", "ESConv.json")
HIEN_DIR = os.path.join(ROOT, "data", "esconv_hien")


SAMPLE_N = int(os.environ.get("CODEMIX_SAMPLE_N", "200"))
STRATEGIES = ["Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
              "Affirmation and Reassurance", "Providing Suggestions", "Information", "Others"]


def load_esconv():
    with open(ESCONV_PATH, encoding="utf-8") as f:
        return json.load(f)


def split(dataset):
    return dataset[:100], dataset[100:]


def hien_dir():
    return os.environ.get("CODEMIX_HIEN_DIR") or HIEN_DIR


def load_version(version):
    if version == "en":
        return split(load_esconv())[0]
    path = os.path.join(hien_dir(), f"test_{version}.json")
    if not os.path.exists(path) and os.environ.get("CODEMIX_ALLOW_PARTIAL"):

        partial = path.replace(".json", ".partial.json")
        if os.path.exists(partial):
            path = partial
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dev_ids():
    with open(os.path.join(hien_dir(), "dev_conv_ids.json"), encoding="utf-8") as f:
        return json.load(f)


def load_dev(version):
    if version == "en":
        data = load_esconv()
        return [dict(data[i], esconv_index=i) for i in dev_ids()]
    with open(os.path.join(hien_dir(), f"dev_{version}.json"), encoding="utf-8") as f:
        return json.load(f)


def dev_samples(version):
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
        if "dialog" in conv:
            out.extend(turn_samples(conv, i))
    return out


def sampled_uids(n=None, seed=42):
    n = n or SAMPLE_N
    uids = [s["uid"] for s in all_samples("en")]
    rnd = random.Random(seed)
    return sorted(rnd.sample(uids, n), key=lambda u: tuple(int(x) for x in u.split("-")))


def case_bank(dataset=None, exclude_conv=()):
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
