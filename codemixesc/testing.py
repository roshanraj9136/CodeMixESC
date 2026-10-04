"""Stand-ins for the heavy components, used by the unit tests and by `--dry_run`.

They exercise every code path of the pipeline without model downloads or API calls:
- LexiconProfiler: the real Profiler logic (tokenisation, chunking, CMI, register profile)
  with HingBERT-LID replaced by a Hinglish word list.
- HashRetriever: the real Retriever logic over a tiny bag-of-words encoder.
- FakeLLM: deterministic, format-following answers to every prompt of the pipeline.
Nothing here is used to produce reported results.
"""
import hashlib
import json
import random
import re
import threading

import numpy as np

from .profiler import Profiler

HINDI_WORDS = set("""
aa aaj aap aapka aapki aapko aaram aata aati aaya ab abhi accha acha achha agar aisa aise ajeeb akela akeli
alag apna apne apni baat baatcheet baatein bachpan bada bade badi bahut bas bata batana batao bataun bhai bhi
bhut bilkul bohot bura buri chahiye chahta chahti chal chala chalo chinta dard dekh dekho dhyan dikkat dil
dimaag dono dost dosti dukh dukhi ek gaya gaye gayi ghabrahat ghar haan hai hain hamesha ho hoga hogi hona
hoon hota hoti hu hua hui hum hun idhar inka inki inko isliye iss itna itni jaana jaanta jaanti jab jaise
jaldi jo jyada ka kaam kab kabhi kaha kahaan kahin kaisa kaise kar kara karna karne karo karta karte karti
karun ke ki kisi kitna ko koi kr kuch kya kyun kyunki lag laga lagta lagti lekin liye logon mai mann mein
mera mere meri mujh mujhe na nahi nahin naukri ne nhi pareshan pata pe pehle phir pyaar raha rahe rahi rakh
rakho sab sach sahi sakta sakte sakti samajh samajhta samajhti se sirf soch socha sochna subah tak tera tere
teri tha theek thi thoda thodi tu tum tumhara tumhari tumhe tumko vo wala wale wali waqt wo woh ya yaar yahan
ye yeh zaroor zarur zindagi
""".split())


# English homographs (is, the, main, hi, par, log, ...) are left out on purpose.


class LexiconProfiler(Profiler):
    """Profiler with the HingBERT-LID call replaced by a word list (tests and dry runs only)."""

    def __init__(self, batch_size=64):  # deliberately skips Profiler.__init__ (no model download)
        self.batch_size = batch_size
        self._memo = {}
        self._lock = threading.Lock()

    def _label_chunks(self, chunks):
        return [["HI" if w.lower() in HINDI_WORDS else "EN" for w in words] for words in chunks]


class HashEncoder:
    """Bag-of-words hashing encoder with the SentenceTransformer.encode() signature."""

    def __init__(self, dim=256):
        self.dim = dim

    def encode(self, texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", (t or "").lower()):
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        if normalize_embeddings:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


class HashRetriever:
    """codemixesc.retriever.Retriever over the HashEncoder (no model download)."""

    def __new__(cls, bank=None, exclude_conv=()):
        from .esconv import case_bank
        from .retriever import Retriever
        r = Retriever.__new__(Retriever)
        r.name, r.model, r.lock = "hash", HashEncoder(), threading.Lock()
        r.bank = bank if bank is not None else case_bank(exclude_conv=set(exclude_conv))
        r.emb = r.encode([b["post"] for b in r.bank])
        return r


_FILLERS = ["yaar", "bahut", "sach mein", "pata nahi", "kya karun", "theek hai", "matlab", "na"]


def fake_hinglish(text, level, seed=0):
    """Deterministic pseudo-Hinglish (Hindi fillers inserted between words) for dry runs."""
    rnd = random.Random(f"{level}-{seed}-{text}")
    rate = {"light": 0.15, "heavy": 0.6}[level]
    out = []
    for w in text.split():
        out.append(w)
        if rnd.random() < rate:
            out.append(rnd.choice(_FILLERS))
    return " ".join(out)


def write_fake_hien(out_dir, levels=("light", "heavy"), n_test=100):
    """Writes test_{level}.json / dev_{level}.json / dev_conv_ids.json in the ESConv-HiEn format
    with pseudo-Hinglish text, so every script can be dry-run without the real dataset."""
    import copy
    import json
    import os
    from .esconv import load_esconv
    data = load_esconv()
    dev = [198, 218, 248, 292, 408, 539, 763, 848, 908, 1139, 1197, 1293]
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dev, open(os.path.join(out_dir, "dev_conv_ids.json"), "w"))
    for level in levels:
        for split, convs in (("test", list(enumerate(data[:n_test]))), ("dev", [(i, data[i]) for i in dev])):
            res = []
            for ci, conv in convs:
                c = copy.deepcopy(conv)
                for k, t in enumerate(c["dialog"]):
                    t["content_en"] = t["content"]
                    t["content"] = fake_hinglish(t["content"].strip(), level, seed=f"{ci}-{k}")
                c["mix_level"], c["esconv_index"] = level, ci
                res.append(c)
            with open(os.path.join(out_dir, f"{split}_{level}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False)
    return out_dir


class FakeLLM:
    """Deterministic stand-in for codemixesc.llm.LLM. It answers every step of the pipeline in
    the format the step asks for (sometimes with markdown, as real models do), choosing among
    the options found in the prompt, so all control-flow branches get exercised."""

    def __init__(self, model="fake", fail_tags=None):
        import os
        self.model = model
        self.n_calls = 0
        # steps whose calls fail like a blocked/empty answer (tests of the retry logic); also
        # settable for a subprocess through CODEMIX_FAKE_FAIL="decide,single"
        env = os.environ.get("CODEMIX_FAKE_FAIL", "")
        self.fail_tags = set(fail_tags if fail_tags is not None else [t for t in env.split(",") if t])

    def __call__(self, prompt, system=None, **kw):
        return self.chat([{"role": "user", "content": prompt}], system=system, **kw)[0]

    def chat(self, messages, system=None, temperature=0.0, max_tokens=400, tag="", json_mode=False):
        self.n_calls += 1
        prompt = messages[0]["content"]
        h = int(hashlib.md5((str(system) + json.dumps(messages)).encode("utf-8")).hexdigest(), 16)
        kind = re.sub(r"\d+$", "", tag)
        if kind in self.fail_tags:
            return "", {"cached": False, "latency": 0.01, "calls": 3, "failed": True, "finish": "SAFETY"}
        agent = int(re.search(r"(\d+)$", tag).group(1)) if re.search(r"\d+$", tag) else 0
        mixed = "Hindi-English mix" in prompt and "plain English" not in prompt
        say = self._hinglish if mixed else self._english
        text = getattr(self, "_" + kind, lambda *a: "Response: " + say(h))(prompt, system, h, agent, say)
        return text, {"cached": False, "latency": 0.01, "calls": 1}

    @staticmethod
    def _english(h):
        return ["I hear you, that sounds really hard.", "What do you think would help you right now?",
                "It makes sense that you feel worried about this.", "Maybe a short walk could clear your mind."][h % 4]

    @staticmethod
    def _hinglish(h):
        return ["Main samajh sakta hoon yaar, yeh sach mein bahut mushkil hai.",
                "Aapko abhi kya help karega, kuch socha hai?",
                "It makes sense ki aap itna pareshan ho, yaar.",
                "Shayad ek chhoti si walk se dimaag thoda halka hoga."][h % 4]

    def _decide(self, prompt, system, h, agent, say):
        return "1. YES\n2. The conversation reflects all three." if h % 5 else "1. NO\n2. The coping plan is missing."

    def _emotion(self, prompt, system, h, agent, say):
        return "**Emotion:** anxiety\n**Reasoning:** The user worries about the future."

    def _cause(self, prompt, system, h, agent, say):
        return "Event: trouble at work\nReasoning: The user mentions their job."

    def _intention(self, prompt, system, h, agent, say):
        return "Intention: to feel understood and find a way forward\nReasoning: ..."

    def _deliberate(self, prompt, system, h, agent, say):
        seen = []
        for s in re.findall(r"\[([A-Za-z][A-Za-z &\-]*)\]", prompt.split("### Examples", 1)[-1]):
            if s not in seen and s != "strategy":
                seen.append(s)
        pick = seen[(h + agent) % len(seen)] if seen else "Question"
        return f"Strategy: [{pick}]\nReasoning: It fits the user's state."

    def _generate(self, prompt, system, h, agent, say):
        m = re.search(r"using the (.+?) strategy", prompt)
        return f"Response: [{m.group(1) if m else 'Question'}] {say(h)}"

    def _supported(self, system):
        m = re.search(r'support the response "\[([^\]]+)\] (.*?)"\. However', system or "", re.S)
        return (m.group(1), m.group(2)) if m else ("Question", "How are you?")

    def _debate(self, prompt, system, h, agent, say):
        s, r = self._supported(system)
        return f"**Response:** [{s}] {r}\n**Reasoning:** I still prefer it."

    def _reflect(self, prompt, system, h, agent, say):
        if h % 7 == 0:  # occasional disagreement -> tie -> judge
            s, r = self._supported(system)
        else:
            disc = prompt.split("### Discussion content", 1)[-1].split("You should carefully analyze", 1)[0]
            m = re.search(r"Response:\**\s*\[([^\]]+)\]\s*(.*)", disc)
            s, r = (m.group(1), m.group(2).strip()) if m else self._supported(system)
        return f"Response: [{s}] {r}\nReasoning: After the discussion this is best."

    def _judge(self, prompt, system, h, agent, say):
        m = re.search(r"\[([^\]]+)\] (.*)", prompt.split("### Examples", 1)[-1])
        return f"Response: [{m.group(1)}] {m.group(2)}\nReasoning: most appropriate." if m else "unclear"

    def _refine(self, prompt, system, h, agent, say):
        m = re.search(r"### Response\n\[([^\]]+)\] (.*)", prompt)
        return f"Response: [{m.group(1)}] {m.group(2)}\nReasoning: It already meets the requirements." if m else "?"

    def _gate(self, prompt, system, h, agent, say):
        m = re.search(r"### Response\n\[([^\]]+)\]", prompt)
        tag = f"[{m.group(1)}] " if m else ""
        return f"Response: {tag}{self._hinglish(h) if 'plain English' not in prompt else self._english(h)}"

    def _translate_in(self, prompt, system, h, agent, say):
        items = json.loads(re.search(r"Conversation:\n(\[.*?\])\n\nReturn", prompt, re.S).group(1))
        return "```json\n" + json.dumps({"turns": [{"id": it["id"], "text": "[en] " + it["text"]} for it in items]}) + "\n```"

    def _translate_out(self, prompt, system, h, agent, say):
        return "Response: " + self._hinglish(h)

    def _fewshot(self, prompt, system, h, agent, say):
        s = ["Question", "Reflection of feelings", "Providing Suggestions"][h % 3]
        return f"Emotion: sadness\nEvent: a breakup\nIntention: to cope\nStrategy: {s}\nResponse: {self._english(h)}"
