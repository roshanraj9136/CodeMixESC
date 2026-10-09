"""Publishes the fine-tuned retriever to the Hugging Face Hub, together with the precomputed
embeddings of the ESConv case bank (vectors only, no text) that the live demo loads.

    python demo/upload_model.py          # after `hf auth login` (write token)
"""
import glob
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK_SIG = "cd1b5da23d"  # md5 prefix of the full ESConv case bank's posts (Retriever._bank_embeddings)

MODEL_CARD = """---
language: [en, hi]
tags: [sentence-transformers, sentence-similarity, code-mixing, hinglish, emotional-support]
library_name: sentence-transformers
base_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
pipeline_tag: sentence-similarity
---
# CodeMixESC cross-lingual retriever

`paraphrase-multilingual-mpnet-base-v2` fine-tuned with the Multiple Negatives Ranking (in-batch
contrastive) loss on 16,951 Hinglish-English sentence pairs (PHINC + Roman-script Hinglish rewrites
of ESConv training utterances), so that a Roman-script Hinglish query retrieves the same English
ESConv cases as its English version. Part of [CodeMixESC](https://github.com/roshanraj9136/CodeMixESC)
(course project, IIT Bhilai); used by its [live demo](https://github.com/roshanraj9136/CodeMixESC#live-demo).

| Dev retrieval (151 turns, k = 10) | Overlap@10 Light | Overlap@10 Heavy |
|---|---|---|
| all-roberta-large-v1 (MultiAgentESC) | 56.0 | 22.1 |
| multilingual mpnet (base) | 69.9 | 29.6 |
| **this model** | **72.9** | **54.0** |

Overlap@10 = share of the cases retrieved for a Hinglish post that are also retrieved for its English
original. Training: 1 epoch, batch 32, lr 2e-5, scale 20, word embeddings frozen; best checkpoint by
dev retrieval. `esconv_bank_embeddings.npy` holds this model's embeddings of the 13,484 ESConv
case-bank posts (no text). For research use.

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("%s")
m.encode(["yaar bahut tension ho rahi hai", "I am very stressed"])
```
"""


def main():
    api = HfApi()
    repo = f"{api.whoami()['name']}/codemix-retriever"
    api.create_repo(repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "m")
        shutil.copytree(os.path.join(ROOT, "models", "codemix-retriever"), dst)
        open(os.path.join(dst, "README.md"), "w", encoding="utf-8").write(MODEL_CARD % repo)
        hits = sorted(glob.glob(os.path.join(ROOT, "cache", "emb", f"mpnet-ft-*-{BANK_SIG}.npy")), key=os.path.getmtime)
        if hits:
            shutil.copy(hits[-1], os.path.join(dst, "esconv_bank_embeddings.npy"))
        api.upload_folder(repo_id=repo, folder_path=dst, commit_message="CodeMixESC retriever + case-bank embeddings")
    print("model:", f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
