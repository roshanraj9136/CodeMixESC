import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc.esconv import ESCONV_PATH, ROOT

BASE_REPO = "https://github.com/MindIntLab-HFUT/MultiAgentESC"
ESCONV_URL = "https://raw.githubusercontent.com/MindIntLab-HFUT/MultiAgentESC/main/dataset/ESConv.json"
MODELS = {
    "sentence": ["sentence-transformers/all-roberta-large-v1", "sentence-transformers/LaBSE",
                 "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"],
    "token": ["l3cube-pune/hing-bert-lid"],
    "bertscore": ["bert-base-multilingual-cased"],
}


def get_base_repo():
    dst = os.path.join(ROOT, "external", "MultiAgentESC")
    if os.path.exists(os.path.join(dst, "multiagent.py")):
        print(f"[setup] base code present: {dst}")
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", BASE_REPO, dst], check=True)
    except Exception as e:
        print(f"[setup] could not clone {BASE_REPO} ({e}); continuing without it")
    return dst


def get_esconv(base_dir):
    if not os.path.exists(ESCONV_PATH):
        os.makedirs(os.path.dirname(ESCONV_PATH), exist_ok=True)
        src = os.path.join(base_dir, "dataset", "ESConv.json")
        if os.path.exists(src):
            shutil.copyfile(src, ESCONV_PATH)
        else:
            print(f"[setup] downloading {ESCONV_URL}")
            urllib.request.urlretrieve(ESCONV_URL, ESCONV_PATH)
    data = json.load(open(ESCONV_PATH, encoding="utf-8"))
    digest = hashlib.sha256(open(ESCONV_PATH, "rb").read()).hexdigest()[:16]
    if len(data) != 1300 or "dialog" not in data[0]:
        sys.exit(f"[setup] unexpected ESConv file ({len(data)} conversations); expected the base repo's 1,300")
    print(f"[setup] ESConv: {len(data)} conversations, sha256 {digest}..., at {ESCONV_PATH}")


def get_models():
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer
    for name in MODELS["sentence"]:
        SentenceTransformer(name, device="cpu")
        print(f"[setup] cached {name}")
    for name in MODELS["token"]:
        AutoTokenizer.from_pretrained(name)
        AutoModelForTokenClassification.from_pretrained(name)
        print(f"[setup] cached {name}")
    for name in MODELS["bertscore"]:
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)
        print(f"[setup] cached {name}")


def main():
    ap = argparse.ArgumentParser(description="One-time setup: the base paper code, ESConv, and optionally the pretrained models.")
    ap.add_argument("--models", action="store_true", help="pre-download all Hugging Face models")
    args = ap.parse_args()
    get_esconv(get_base_repo())
    if args.models:
        get_models()


if __name__ == "__main__":
    main()
