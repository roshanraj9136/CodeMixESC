import hashlib
import html
import os
import re
import shutil
import sys
import threading
import time
import urllib.request

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL = "gemma-4-26b-a4b-it"
RETRIEVER_REPO = "roshan9136/codemix-retriever"
BANK_EMB_FILE = "esconv_bank_embeddings.npy"
ESCONV_URL = "https://raw.githubusercontent.com/MindIntLab-HFUT/MultiAgentESC/main/dataset/ESConv.json"
GITHUB = "https://github.com/roshanraj9136/CodeMixESC"
HF_MODEL = f"https://huggingface.co/{RETRIEVER_REPO}"
ARCH_IMG = os.path.join(ROOT, "demo", "architecture.svg")
LOGO_IMG = os.path.join(ROOT, "demo", "logo.svg")
DELTA = 0.15
MAX_TURNS_PER_DAY = 400

st.set_page_config(page_title="CodeMixESC - Hinglish emotional support",
                   layout="wide", initial_sidebar_state="collapsed")


def api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


@st.cache_resource(show_spinner="Loading the language identifier and the retriever (first visit only)...")
def load_pipeline():
    os.environ.setdefault("CODEMIX_DEVICE", "cpu")
    from codemixesc.esconv import ESCONV_PATH, case_bank
    if not os.path.exists(ESCONV_PATH):
        os.makedirs(os.path.dirname(ESCONV_PATH), exist_ok=True)
        urllib.request.urlretrieve(ESCONV_URL, ESCONV_PATH)
    from codemixesc import retriever as RT
    local = RT.ENCODERS["mpnet-ft"]
    if not os.path.exists(os.path.join(local, "modules.json")):
        from huggingface_hub import snapshot_download
        snapshot_download(RETRIEVER_REPO, local_dir=local)
    bank_sig = hashlib.md5("\n".join(b["post"] for b in case_bank()).encode("utf-8")).hexdigest()[:10]
    dst = os.path.join(RT.EMB_DIR, f"mpnet-ft-{RT.model_signature(local)}-{bank_sig}.npy")
    src = os.path.join(local, BANK_EMB_FILE)
    if not os.path.exists(dst) and os.path.exists(src):
        os.makedirs(RT.EMB_DIR, exist_ok=True)
        shutil.copyfile(src, dst)
    from codemixesc.llm import LLM
    from codemixesc.profiler import Profiler
    from codemixesc.systems import SYSTEMS, System
    retriever = RT.Retriever("mpnet-ft")
    system = System("codemixesc", LLM(MODEL), retriever=retriever, profiler=Profiler(), delta=DELTA,
                    **SYSTEMS["codemixesc"])
    return system, retriever


@st.cache_resource
def usage_counter():
    return {"day": time.strftime("%Y-%m-%d"), "n": 0}


class ProgressLLM:
    def __init__(self, llm, events):
        self.llm, self.events = llm, events

    def chat(self, messages, **kw):
        tag = kw.get("tag", "")
        self.events.append(("start", tag, time.time()))
        try:
            return self.llm.chat(messages, **kw)
        finally:
            self.events.append(("end", tag, time.time()))

    def __call__(self, prompt, system=None, **kw):
        return self.chat([{"role": "user", "content": prompt}], system=system, **kw)[0]


STAGES = {"decide": "Decision maker: is the dialogue complex?",
          "single": "One agent writes the reply",
          "emotion": "Emotion agent",
          "cause": "Cause agent",
          "intention": "Intention agent",
          "deliberate": "Strategy deliberation (3 agents)",
          "generate": "Writing one reply per strategy",
          "debate": "Debate between the candidate replies",
          "reflect": "Reflection and vote",
          "judge": "Judge breaks a tie",
          "refine": "Refiner polishes the winner",
          "gate": "Register Gate rewrites the reply in your mix"}


def stage_lines(events, t0, done=False):
    order, first, last, open_n = [], {}, {}, {}
    for kind, tag, t in list(events):
        stage = re.sub(r"\d+$", "", tag)
        if stage not in STAGES:
            continue
        if stage not in first:
            order.append(stage)
            first[stage] = t
        open_n[stage] = open_n.get(stage, 0) + (1 if kind == "start" else -1)
        if kind == "end":
            last[stage] = t
    lines = ["Code-Mix Profiler measured your language mix"]
    for stage in order:
        if stage == "deliberate":
            lines.append("Cross-lingual retriever found 10 similar past cases")
        label = STAGES[stage]
        if open_n.get(stage, 0) > 0 and not done:
            lines.append(f"{label} ...")
        else:
            secs = last.get(stage, first[stage]) - first[stage]
            lines.append(f"{label} ({secs:.0f} s)")
    if not done and (not order or open_n.get(order[-1], 0) <= 0):
        lines.append(f"thinking... {time.time() - t0:.0f} s")
    return lines


def make_sample(dialog, early_rule):
    from codemixesc.esconv import json2natural
    count = len(dialog) + 1
    return {"uid": f"demo-{len(dialog)}", "conv_id": -1, "turn": len(dialog), "strategy": "", "reference": "",
            "context_msgs": [{"role": m["role"], "content": m["content"]} for m in dialog],
            "context": json2natural(dialog), "post": dialog[-1]["content"],
            "early": early_rule and count <= 5}


def respond_live(dialog, early_rule, status):
    from codemixesc.systems import SYSTEMS, System
    base, retriever = load_pipeline()
    events, out = [], {}
    system = System("codemixesc", ProgressLLM(base.llm, events), retriever=retriever, profiler=base.profiler,
                    delta=DELTA, **SYSTEMS["codemixesc"])

    def work():
        try:
            out["rec"] = system.respond(make_sample(dialog, early_rule), "demo")
        except Exception as e:
            out["err"] = e

    t0 = time.time()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    shown = 0
    while worker.is_alive():
        lines = stage_lines(events, t0)
        for line in lines[shown:]:
            status.write(line)
        shown = len(lines)
        worker.join(0.4)
    lines = stage_lines(events, t0, done=True)
    for line in lines[shown:]:
        status.write(line)
    if "err" in out:
        raise out["err"]
    return out["rec"], lines


def esc(t):
    return html.escape(str(t)).replace("\n", " ").replace("$", "\\$")


def mix_bar_svg(hindi_share):
    h = max(0.0, min(1.0, float(hindi_share)))
    width = 320
    hi_w = round(width * h)
    en_w = width - hi_w
    hi_label = f"{round(100 * h)}% Hindi" if hi_w >= 70 else ""
    en_label = f"{round(100 * (1 - h))}% English" if en_w >= 80 else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} 30" '
        f'preserveAspectRatio="none" role="img" aria-label="Language mix: {round(100 * h)} percent Hindi">'
        f'<rect x="0" y="0" width="{width}" height="30" rx="6" fill="#F4F6FA"/>'
        f'<rect x="0" y="0" width="{hi_w}" height="30" rx="6" fill="#C2410C"/>'
        f'<rect x="{hi_w}" y="0" width="{en_w}" height="30" rx="6" fill="#2F5D9A"/>'
        f'<text x="10" y="20" font-family="Inter, Segoe UI, sans-serif" font-size="13" fill="#FFFFFF">{hi_label}</text>'
        f'<text x="{width - 10}" y="20" text-anchor="end" font-family="Inter, Segoe UI, sans-serif" '
        f'font-size="13" fill="#FFFFFF">{en_label}</text>'
        f'</svg>'
    )


STRATEGY_PLAIN = {
    "Question": "asks a gentle, open question so you can say more",
    "Restatement or Paraphrasing": "says back what you said, more simply, so you can see it clearly",
    "Reflection of feelings": "names the feeling it hears, so you feel understood",
    "Self-disclosure": "shares a similar experience to show it relates",
    "Affirmation and Reassurance": "points out your strengths and gives encouragement",
    "Providing Suggestions": "offers an idea to try, without pushing",
    "Information": "gives useful information or facts",
    "Others": "keeps the conversation warm with greetings, thanks or small talk",
}


def bar(hindi_share):
    st.markdown(mix_bar_svg(hindi_share), unsafe_allow_html=True)


def insight_panel(rec, retriever):
    from codemixesc.profiler import is_plain_english
    R, g = rec.get("R") or {}, rec.get("gate") or {}
    st.markdown("**Your language mix**")
    if R:
        bar(R["hi_frac"])
        if is_plain_english(R):
            st.write("Plain English, Roman script.")
        else:
            st.write(f"Code-mixed Hinglish, CMI {R['cmi']:.2f}, {R.get('script', 'Roman')} script.")
        st.caption("Measured word by word with HingBERT-LID. CMI is how mixed the two languages are "
                   "(0 is one language, 0.5 is half and half).")
    a = rec.get("analysis") or {}
    if rec.get("path") != "single" and a:
        st.markdown("**What it understood**")
        st.write(f"Feeling: {esc(a.get('emotion'))}")
        st.write(f"Because: {esc(a.get('cause'))}")
        st.write(f"What you want: {esc(a.get('intention'))}")
    strat = rec.get("pred_strategy")
    st.markdown("**Support strategy**")
    if strat and strat in STRATEGY_PLAIN:
        st.write(f"{strat}: {STRATEGY_PLAIN[strat]}.")
        cands = rec.get("candidates") or []
        if len(cands) > 1:
            st.caption(f"{len(cands)} replies were written, one per strategy, and the agents debated and "
                       f"voted for this one.")
    else:
        st.write("Early or simple turn: one agent answered directly, as in MultiAgentESC.")
    if g:
        st.markdown("**Register Gate**")
        st.caption("Target mix")
        bar(g.get("hi_frac_target", 0))
        reply_label = "Reply (kept original)" if g.get("triggered") and not g.get("accepted") else "Reply mix"
        st.caption(reply_label)
        bar(g.get("hi_frac_after") if g.get("accepted") else g.get("hi_frac_before", 0))
        if not g.get("triggered"):
            st.write("Matched on the first try.")
        elif g.get("accepted"):
            st.write(f"Rewritten to match (was {100 * g['hi_frac_before']:.0f}% Hindi).")
        else:
            st.write("The rewrite was not closer, so the original was kept.")
    with st.expander(f"Behind the scenes: {rec.get('n_calls', 0)} LLM calls, "
                     f"{rec.get('latency', 0):.0f} s of API time"):
        ids = rec.get("retrieved") or []
        if ids and retriever is not None:
            shown = min(3, len(ids))
            st.write(f"Similar past cases from the English case bank (showing {shown} of {len(ids)}):")
            for i in ids[:shown]:
                b = retriever.bank[i]
                st.write(f"- \"{esc(b['post'][:140])}\" -> [{esc(b['strategy'])}] {esc(b['response'][:120])}")
        if rec.get("strategies"):
            st.write("Strategies proposed by the 3 agents: " + ", ".join(rec["strategies"]))
        for s, r in rec.get("candidates") or []:
            st.write(f"- [{esc(s)}] {esc(r)}")
        v = rec.get("vote") or {}
        if v.get("strategies"):
            st.write("Vote winner: " + esc(", ".join(v["strategies"]))
                     + (" (tie broken by the judge)" if rec.get("judge") else ""))
        st.caption(f"Agents: {MODEL} through the Gemini API, temperature 0.")


def empty_panel():
    st.markdown("**Nothing here yet**")
    st.write("Send a message and this panel will fill in: your language mix, the feeling, the cause and "
             "intention, the chosen strategy and the Register Gate result.")


key = api_key()
if key:
    os.environ["GEMINI_API_KEY"] = key
if "dialog" not in st.session_state:
    st.session_state.dialog = []
pending = None
if st.session_state.dialog and st.session_state.dialog[-1]["role"] == "user":
    pending = st.session_state.dialog.pop()["content"]

head_a, head_b = st.columns([1, 11], vertical_alignment="center")
with head_a:
    st.image(LOGO_IMG, width=46)
with head_b:
    st.title("CodeMixESC")
st.write("Emotional support that answers in your own Hindi-English mix. Write in English or Roman-script "
         "Hinglish, and the panel on the right shows what the agents understood.")

top_l, top_r = st.columns([5, 2], vertical_alignment="center")
with top_l:
    st.caption("A course project (IIT Bhilai) extending MultiAgentESC (EMNLP 2025) to code-mixed Hinglish "
               "help-seekers.")
with top_r:
    early_rule = st.toggle("First turns use one agent", value=False,
                           help="MultiAgentESC answers the first turns with a single agent. Off means the full "
                                "pipeline from the first message.")
    new_chat = st.button("New chat")
if new_chat:
    st.session_state.dialog = []
    st.rerun()

if not key:
    st.error("GEMINI_API_KEY is not configured for this app.")
    st.stop()

chat_col, info_col = st.columns([3, 2], gap="large")
SUGGESTIONS = {
    "Exam result stress": "yaar exams ka result aaya hai aur bahut tension ho rahi hai, ghar pe kaise bataun "
                          "samajh nahi aa raha",
    "No job yet": "Mujhe lagta hai main kabhi job nahi dhoond paunga, sab friends settle ho gaye aur main abhi "
                  "bhi wahi hoon",
    "Friend stopped talking": "meri best friend ne mujhse baat karna band kar diya, I don't even know what I "
                              "did wrong",
    "In English": "I lost my job last month and honestly I feel useless",
}

with chat_col:
    box = st.container(height=330, border=True)
    with box:
        if not st.session_state.dialog:
            with st.chat_message("assistant"):
                st.markdown("Hi, I'm here to listen. Tell me what's on your mind, in English, Hindi or Hinglish, "
                            "whatever feels natural. This is a research prototype, not a crisis service.")
        for msg in st.session_state.dialog:
            with st.chat_message(msg["role"]):
                rec = msg.get("rec")
                if msg.get("steps") and rec:
                    with st.expander(f"Answered in {msg['secs']:.0f} s, {rec['n_calls']} LLM calls, see every step"):
                        for line in msg["steps"]:
                            st.write(line)
                st.markdown(esc(msg["content"]))
                if rec and rec.get("pred_strategy") in STRATEGY_PLAIN:
                    st.caption(rec["pred_strategy"])
    picked = None
    if not st.session_state.dialog:
        st.caption("Try an example:")
        cols = st.columns(len(SUGGESTIONS))
        for col, (label, text) in zip(cols, SUGGESTIONS.items()):
            if col.button(label):
                picked = text
    prompt = st.chat_input("Type in English or Hinglish, e.g. yaar bahut tension ho rahi hai...")
    st.caption("Not a crisis or counselling service. In distress in India? Call Tele-MANAS 14416 (24x7). "
               "Elsewhere, contact your local emergency or crisis service.")

with info_col:
    st.markdown("### What it understood")
    panel = st.empty()

with st.expander("How it works"):
    if os.path.exists(ARCH_IMG):
        st.image(ARCH_IMG)
    st.caption("Orange is new in CodeMixESC, dashed is modified, blue is MultiAgentESC.")
    st.write("1. The Code-Mix Profiler measures your language mix word by word.")
    st.write("2. Agents read the message for feeling, cause and intention.")
    st.write("3. The cross-lingual retriever recalls similar past cases.")
    st.write("4. One reply is written per strategy, then the agents debate and vote.")
    st.write("5. The Register Gate rewrites the winner to match your Hinglish.")

st.caption(f"[Source code]({GITHUB}) ? [Fine-tuned retriever]({HF_MODEL}) ? Roshan Raj ? 12341830")


def draw_panel(rec=None):
    with panel.container():
        if rec:
            insight_panel(rec, load_pipeline()[1])
        else:
            empty_panel()


prompt = (prompt or picked or pending or "").strip()
if not prompt:
    draw_panel(next((m.get("rec") for m in reversed(st.session_state.dialog) if m.get("rec")), None))
if prompt:
    counter = usage_counter()
    if counter["day"] != time.strftime("%Y-%m-%d"):
        counter.update(day=time.strftime("%Y-%m-%d"), n=0)
    st.session_state.dialog.append({"role": "user", "content": prompt})
    with box:
        with st.chat_message("user"):
            st.markdown(esc(prompt))
        with st.chat_message("assistant"):
            steps, t0 = None, time.time()
            if counter["n"] >= MAX_TURNS_PER_DAY:
                rec, reply = None, "The demo reached today's free-tier limit; please try again tomorrow."
            else:
                counter["n"] += 1
                with st.status("The agents are working on your reply...", expanded=True) as status:
                    t0 = time.time()
                    try:
                        rec, steps = respond_live(st.session_state.dialog, early_rule, status)
                        reply = rec["response"]
                        status.update(label=f"Answered in {time.time() - t0:.0f} s, {rec['n_calls']} LLM calls",
                                      state="complete", expanded=False)
                    except Exception as e:
                        rec, reply = None, "Sorry, the model is busy right now. Please send your message again."
                        status.update(label=f"Model busy ({type(e).__name__})", state="error")
            st.markdown(esc(reply))
            if rec and rec.get("pred_strategy") in STRATEGY_PLAIN:
                st.caption(rec["pred_strategy"])
    st.session_state.dialog.append({"role": "assistant", "content": reply, "rec": rec, "steps": steps,
                                    "secs": time.time() - t0})
    st.rerun()
