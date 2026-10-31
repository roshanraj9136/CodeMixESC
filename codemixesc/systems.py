import random
import re

from . import agents as A
from . import prompts as P
from .esconv import dev_ids, json2natural, load_esconv, turn_samples
from .profiler import is_plain_english

SYSTEMS = {
    "zero_shot": dict(kind="zero_shot"),
    "fewshot_cot": dict(kind="fewshot_cot"),
    "maesc": dict(kind="multi", encoder="roberta", register=False, gate=False),
    "pivot": dict(kind="pivot", encoder="roberta"),
    "codemixesc": dict(kind="multi", encoder="mpnet-ft", register=True, gate=True),
    "cmx_noft": dict(kind="multi", encoder="mpnet", register=True, gate=True),
    "cmx_noxl": dict(kind="multi", encoder="roberta", register=True, gate=True),
}
FEWSHOT_TARGETS = ["Question", "Reflection of feelings", "Affirmation and Reassurance", "Providing Suggestions"]


def needs(name_or_spec):
    spec = SYSTEMS[name_or_spec] if isinstance(name_or_spec, str) else name_or_spec
    enc = spec.get("encoder") if spec["kind"] in ("multi", "pivot") else None
    return enc, bool(spec.get("register") or spec.get("gate") or spec["kind"] == "pivot")


def dev_conv_ids():
    try:
        return set(dev_ids())
    except FileNotFoundError:
        return set()


def fewshot_examples(dataset=None):
    data = dataset or load_esconv()
    order = list(range(len(data)))
    random.Random(42).shuffle(order)
    banned = set(range(100)) | dev_conv_ids()
    picked = {}
    for ci in order[400:420] + order[420:] + order[:400]:
        if ci in banned:
            continue
        conv = data[ci]
        for s in turn_samples(conv, ci):
            n = len(s["reference"].split())
            if s["early"] or s["strategy"] not in FEWSHOT_TARGETS or s["strategy"] in picked or not 8 <= n <= 30:
                continue
            picked[s["strategy"]] = P.FEWSHOT_EXAMPLE.format(
                context=json2natural(s["context_msgs"][-6:]), emotion=conv["emotion_type"],
                event=conv["situation"].strip(), intention=f"to find a way to cope with {conv['problem_type'].lower()}",
                strategy=s["strategy"], response=s["reference"])
        if len(picked) == len(FEWSHOT_TARGETS):
            break
    return "\n\n".join(picked[t] for t in FEWSHOT_TARGETS if t in picked)


class System:
    def __init__(self, name, llm, retriever=None, profiler=None, delta=0.2, **overrides):
        spec = dict(SYSTEMS.get(name, {}))
        spec.update(overrides)
        self.name, self.spec, self.kind = name, spec, spec["kind"]
        self.llm, self.retriever, self.profiler = llm, retriever, profiler
        self.register = bool(spec.get("register"))
        self.delta = delta if spec.get("gate") else None
        enc, needs_profiler = needs(spec)
        if enc and retriever is None:
            raise ValueError(f"{name} needs a retriever ({enc})")
        if needs_profiler and profiler is None:
            raise ValueError(f"{name} needs the Code-Mix Profiler")
        self._fewshot = fewshot_examples() if self.kind == "fewshot_cot" else None

    def respond(self, sample, version):
        log = A.CallLog(self.llm)
        rec = {"uid": sample["uid"], "conv_id": sample["conv_id"], "turn": sample["turn"], "version": version,
               "system": self.name, "early": sample["early"], "gold_strategy": sample["strategy"],
               "reference": sample["reference"], "post": sample["post"]}
        R = None
        if self.profiler is not None:
            R = self.profiler.profile([m["content"] for m in sample["context_msgs"] if m["role"] == "user"])
        rec["R"] = R
        if self.kind == "zero_shot":
            resp, raw = A.single_agent_response(log, sample["context"])
            rec.update(path="single", complex=None, pred_strategy="None", raw=raw)
            rec["pre_gate_response"] = rec["response"] = A.clean_response(resp)
            rec["gate"] = None
        elif self.kind == "fewshot_cot":
            rec.update(self._fewshot_turn(log, sample))
        elif self.kind == "multi":
            rec.update(self._multiagent_turn(log, sample, R if self.register else None, R))
        elif self.kind == "pivot":
            rec.update(self._pivot_turn(log, sample, R))
        else:
            raise ValueError(self.kind)
        rec.update(log.summary())
        return rec

    def _fewshot_turn(self, log, sample):
        prompt = P.FEWSHOT_COT.format(strategy_definitions=P.strategy_definitions_text(), examples=self._fewshot,
                                      context=sample["context"])
        raw = log.ask(prompt, max_tokens=A.ANALYSIS_MAX_TOKENS, tag="fewshot")
        strategy = A.parse_strategies(raw)

        resp = A.clean_response(A.parse_single(raw)) if re.search(r"Response\s*:", A.plain(raw)) else "None"
        return {"path": "single", "complex": None, "pred_strategy": strategy[0] if strategy else "None", "raw": raw,
                "pre_gate_response": resp, "response": resp, "gate": None}

    def _multiagent_turn(self, log, sample, R_prompt, R_gate):
        out = {}
        context, post = sample["context"], sample["post"]
        if sample["early"]:
            out["complex"] = None
        else:
            out["complex"], out["decide_raw"] = A.is_complex(log, context)
        strategy, response = "None", None
        if sample["early"] or not out["complex"]:
            response, _ = A.single_agent_response(log, context, R_prompt)
            out["path"] = "single"
        else:
            an = A.analyse(log, context, R_prompt)
            strategies, pairs, idx, delib = A.get_strategy(log, self.retriever, context, an, post)
            out.update(analysis={k: an[k] for k in ("emotion", "cause", "intention")}, retrieved=idx,
                       deliberation=delib, strategies=strategies)
            cands = []
            for s in strategies:
                r, _ = A.response_with_strategy(log, context, an, s, A.examples_for(pairs, s), R_prompt)
                cands.append([s, r])
            out["candidates"] = cands
            valid = [(s, r) for s, r in cands if A.usable(r)]
            if not valid:
                response, _ = A.single_agent_response(log, context, R_prompt, tag="fallback")
                out["path"] = "fallback"
            else:
                out["path"] = "multi"
                if len(valid) == 1:
                    strategy, response = valid[0]
                else:
                    tagged = [f"[{s}] {r}" for s, r in valid]
                    deb = A.debate(log, context, an, tagged, R_prompt)
                    ref = A.reflect(log, context, an, deb, tagged, R_prompt)
                    strats, resps = A.vote(ref, valid)
                    out.update(debate=deb, reflection=ref, vote={"strategies": strats, "responses": resps})
                    if len(strats) == 1 and strats[0] != "None":
                        strategy, response = strats[0], resps[0]
                        out["judge"] = False
                    else:
                        pool = valid if strats == ["None"] else list(zip(strats, resps))
                        strategy, response, _ = A.judge(log, context, pool)
                        out["judge"] = True
                out["ori_response"] = response
                response, _ = A.self_reflection(log, context, strategy, response, R_prompt)
                out["refined_response"] = response
        response = A.clean_response(response)
        out["pred_strategy"] = strategy
        out["pre_gate_response"] = response
        out["gate"] = None
        if self.delta is not None and R_gate is not None:
            response, out["gate"] = A.register_gate(log, self.profiler, context, R_gate, strategy, response, self.delta,
                                                    self.spec.get("gate_metric", "hi_frac"))
        out["response"] = A.clean_response(response)
        return out

    def _pivot_turn(self, log, sample, R):
        msgs_en, complete = A.translate_context(log, sample["context_msgs"])
        en = dict(sample, context_msgs=msgs_en, context=json2natural(msgs_en), post=msgs_en[-1]["content"])
        out = self._multiagent_turn(log, en, None, None)
        response_en = out["response"]
        response = response_en if is_plain_english(R) else A.translate_response(log, response_en, R)
        out["pivot"] = {"context_en": en["context"], "post_en": en["post"], "response_en": response_en,
                        "translation_complete": complete}
        out["pre_gate_response"] = out["response"] = A.clean_response(response)
        return out
