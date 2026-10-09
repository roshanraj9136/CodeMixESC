"""CodeMixESC live demo (Streamlit).

Write as a help-seeker in English or Roman-script Hinglish. CodeMixESC's multi-agent pipeline
answers in your own Hindi-English mix, and the panel on the right shows what the agents
understood: your language mix, emotion, cause and intention, the support strategy they chose
and the Register Gate's language check. "Behind the scenes" lists the retrieved cases, the
candidate replies and the vote.

Local run (from the repository root):
    streamlit run demo/streamlit_app.py
Needs GEMINI_API_KEY (a free AI Studio key) in .streamlit/secrets.toml, the environment or .env.
"""
import hashlib
import html
import os
import shutil
import sys
import time
import urllib.request

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL = "gemma-4-26b-a4b-it"
RETRIEVER_REPO = "roshan9136/codemix-retriever"
BANK_EMB_FILE = "esconv_bank_embeddings.npy"  # precomputed case-bank vectors in the model repo
ESCONV_URL = "https://raw.githubusercontent.com/MindIntLab-HFUT/MultiAgentESC/main/dataset/ESConv.json"
GITHUB = "https://github.com/roshanraj9136/CodeMixESC"
ARCH_IMG = "https://raw.githubusercontent.com/roshanraj9136/CodeMixESC/master/report/figures/architecture_slide.png"
DELTA = 0.15  # tuned on the development set (results/tuning/delta.json)
MAX_TURNS_PER_DAY = 400

st.set_page_config(page_title="CodeMixESC · Hinglish emotional support", page_icon=":material/favorite:",
                   layout="wide", initial_sidebar_state="collapsed")


# ------------------------------------------------------------------ backend
def api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:  # no secrets file
        pass
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


@st.cache_resource(show_spinner="Waking up the agents: loading the language identifier and the retriever (first visit only)…")
def load_pipeline():
    """ESConv (downloaded from the base paper's repository, not redistributed), the fine-tuned
    retriever and its precomputed case-bank embeddings (Hugging Face Hub), the profiler and the
    LLM client; shared by all sessions."""
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
    # the precomputed vectors under the name Retriever._bank_embeddings() looks for
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


def respond(dialog, early_rule):
    from codemixesc.esconv import json2natural
    count = len(dialog) + 1  # dialog index after this supporter turn, as in the base main.py
    sample = {"uid": f"demo-{len(dialog)}", "conv_id": -1, "turn": len(dialog), "strategy": "", "reference": "",
              "context_msgs": list(dialog), "context": json2natural(dialog), "post": dialog[-1]["content"],
              "early": early_rule and count <= 5}
    system, _ = load_pipeline()
    return system.respond(sample, "demo")


def esc(t):
    return html.escape(str(t)).replace("\n", " ").replace("$", "\\$")


STRATEGY_ICON = {"Question": ":material/help:", "Restatement or Paraphrasing": ":material/repeat:",
                 "Reflection of feelings": ":material/favorite:", "Self-disclosure": ":material/person:",
                 "Affirmation and Reassurance": ":material/thumb_up:", "Providing Suggestions": ":material/lightbulb:",
                 "Information": ":material/info:", "Others": ":material/chat:"}
STRATEGY_PLAIN = {
    "Question": "asks a gentle, open question so you can say more",
    "Restatement or Paraphrasing": "says back what you said, more simply, so you can see it clearly",
    "Reflection of feelings": "names the feeling it hears, so you feel understood",
    "Self-disclosure": "shares a similar experience to show it relates",
    "Affirmation and Reassurance": "points out your strengths and gives encouragement",
    "Providing Suggestions": "offers an idea to try, without pushing",
    "Information": "gives useful information or facts",
    "Others": "keeps the conversation warm (greetings, thanks, small talk)",
}


# ------------------------------------------------------------------ insight panel
def mix_bar(label, hindi_share):
    h = max(0.0, min(1.0, float(hindi_share)))
    st.progress(h, text=f"{label}: **{100 * h:.0f}% Hindi** · {100 * (1 - h):.0f}% English")


def insight_panel(rec, retriever):
    from codemixesc.profiler import is_plain_english
    R, g = rec.get("R") or {}, rec.get("gate") or {}
    with st.container(border=True):
        st.markdown("##### :material/translate: Your language mix")
        if R:
            mix_bar("You", R["hi_frac"])
            with st.container(horizontal=True):
                if is_plain_english(R):
                    st.badge("Plain English", icon=":material/language:", color="gray")
                else:
                    st.badge(f"Code-mixed Hinglish · CMI {R['cmi']:.2f}", icon=":material/merge_type:", color="orange")
                st.badge(f"{R.get('script', 'Roman')} script", icon=":material/keyboard:", color="blue")
            st.caption("Measured word by word with HingBERT-LID; CMI = how mixed the two languages are (0 = one "
                       "language, 0.5 = half-half).")
    a = rec.get("analysis") or {}
    if rec.get("path") != "single" and a:
        with st.container(border=True):
            st.markdown("##### :material/psychology: What the agents understood")
            st.markdown(f":material/mood: **Feeling** — {esc(a.get('emotion'))}")
            st.markdown(f":material/event: **Because** — {esc(a.get('cause'))}")
            st.markdown(f":material/flag: **What you want** — {esc(a.get('intention'))}")
    strat = rec.get("pred_strategy")
    with st.container(border=True):
        st.markdown("##### :material/volunteer_activism: Support strategy")
        if strat and strat in STRATEGY_PLAIN:
            st.badge(strat, icon=STRATEGY_ICON.get(strat), color="violet")
            st.caption(f"The reply {STRATEGY_PLAIN[strat]}.")
            cands = rec.get("candidates") or []
            if len(cands) > 1:
                st.caption(f"{len(cands)} candidate replies were written (one per strategy) and the agents debated "
                           f"and voted for this one.")
        else:
            st.caption("Early or simple turn: one agent answered directly (as in MultiAgentESC).")
    if g:
        with st.container(border=True):
            st.markdown("##### :material/rule: Register Gate · language check")
            mix_bar("Target", g.get("hi_frac_target", 0))
            mix_bar("Reply", g.get("hi_frac_after") if g.get("accepted") else g.get("hi_frac_before", 0))
            if not g.get("triggered"):
                st.badge("Matched on the first try", icon=":material/check_circle:", color="green")
            elif g.get("accepted"):
                st.badge(f"Rewritten to match (was {100 * g['hi_frac_before']:.0f}% Hindi)",
                         icon=":material/autorenew:", color="orange")
            else:
                st.badge("Rewrite was not closer, original kept", icon=":material/info:", color="gray")
    with st.expander(f":material/manufacturing: Behind the scenes · {rec.get('n_calls', 0)} LLM calls "
                     f"({rec.get('latency', 0):.0f} s of model time, partly in parallel)"):
        ids = rec.get("retrieved") or []
        if ids:
            st.markdown("**Similar past cases** found by the cross-lingual retriever (3 of 10, English case bank):")
            for i in ids[:3]:
                b = retriever.bank[i]
                st.markdown(f"- “{esc(b['post'][:140])}” → *[{esc(b['strategy'])}]* {esc(b['response'][:120])}")
        if rec.get("strategies"):
            st.markdown("**Strategies proposed** by the 3-agent group chat: "
                        + ", ".join(f"`{s}`" for s in rec["strategies"]))
        for s, r in rec.get("candidates") or []:
            st.markdown(f"- *[{esc(s)}]* {esc(r)}")
        v = rec.get("vote") or {}
        if v.get("strategies"):
            st.markdown(f"**Vote winner:** {esc(', '.join(v['strategies']))}"
                        + (" (tie broken by the judge)" if rec.get("judge") else ""))
        st.caption(f"Agents: `{MODEL}` via the Gemini API (free tier), temperature 0.")


def empty_panel():
    with st.container(border=True):
        st.markdown("##### :material/auto_awesome: How it works")
        steps = [(":material/translate:", "Measures your language mix", "word-by-word Hindi/English tagging"),
                 (":material/psychology:", "Understands you", "agents for feeling, cause and intention"),
                 (":material/travel_explore:", "Recalls similar cases", "retriever trained on Hinglish–English pairs"),
                 (":material/forum:", "Debates the best reply", "one reply per strategy, then a vote"),
                 (":material/rule:", "Checks the language", "the Register Gate matches your Hinglish")]
        for icon, head, sub in steps:
            st.markdown(f"{icon} **{head}**  \n:gray[{sub}]")
    st.caption("Send a message, and this panel will show what the agents understood.")


# ------------------------------------------------------------------ page
key = api_key()
if key:
    os.environ["GEMINI_API_KEY"] = key
if "dialog" not in st.session_state:
    st.session_state.dialog, st.session_state.traces = [], []

with st.sidebar:
    st.markdown("### CodeMixESC")
    st.markdown("A course project (IIT Bhilai) extending **MultiAgentESC** (EMNLP 2025) to code-mixed "
                "Hinglish help-seekers.")
    st.markdown(f"[:material/code: Code, data and results]({GITHUB})  \n"
                f"[:material/neurology: Fine-tuned retriever](https://huggingface.co/{RETRIEVER_REPO})")
    early_rule = st.toggle("Base-paper rule: first turns use one agent", value=False,
                           help="MultiAgentESC answers the first turns of a dialogue with a single zero-shot "
                                "agent. Off = the full pipeline from the first message.")
    st.caption("Roshan Raj · 12341830")

head, side = st.columns([5, 2], vertical_alignment="center")
with head:
    st.title("CodeMixESC")
    st.markdown("##### Emotional support that understands how you really talk — English, Hindi, or both.")
    with st.container(horizontal=True):
        st.badge("8 cooperating AI agents", icon=":material/groups:", color="orange")
        st.badge("Understands Hinglish", icon=":material/translate:", color="blue")
        st.badge("Replies in your own mix", icon=":material/tune:", color="green")
with side:
    with st.container(horizontal=True, horizontal_alignment="right"):
        with st.popover("How it works", icon=":material/account_tree:"):
            st.image(ARCH_IMG, alt="CodeMixESC architecture: profiler, three agent stages, cross-lingual "
                                   "retriever and Register Gate")
            st.caption("Orange = new in CodeMixESC, dashed = modified, blue = MultiAgentESC.")
        if st.button("New chat", icon=":material/refresh:"):
            st.session_state.dialog, st.session_state.traces = [], []
            st.rerun()

if not key:
    st.error("GEMINI_API_KEY is not configured for this app.", icon=":material/key_off:")
    st.stop()

chat_col, info_col = st.columns([3, 2], gap="large")
SUGGESTIONS = {
    ":material/school: Exam result stress": "yaar exams ka result aaya hai aur bahut tension ho rahi hai, "
                                            "ghar pe kaise bataun samajh nahi aa raha",
    ":material/work: No job yet": "Mujhe lagta hai main kabhi job nahi dhoond paunga, sab friends settle ho gaye "
                                  "aur main abhi bhi wahi hoon",
    ":material/group: Friend stopped talking": "meri best friend ne mujhse baat karna band kar diya, "
                                               "I don't even know what I did wrong",
    ":material/language: In English": "I lost my job last month and honestly I feel useless",
}

with chat_col:
    box = st.container(height=330, border=True)
    with box:
        if not st.session_state.dialog:
            with st.chat_message("assistant", avatar=":material/favorite:"):
                st.markdown("Hi, I'm here to listen. Tell me what's on your mind — in English, Hindi or Hinglish, "
                            "whatever feels natural. *(Research prototype, not a crisis service.)*")
        for i, msg in enumerate(st.session_state.dialog):
            avatar = ":material/favorite:" if msg["role"] == "assistant" else ":material/person:"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(esc(msg["content"]))
                if msg["role"] == "assistant":
                    rec = st.session_state.traces[i // 2]
                    if rec and rec.get("pred_strategy") in STRATEGY_PLAIN:
                        st.caption(f"{STRATEGY_ICON[rec['pred_strategy']]} {rec['pred_strategy']}")
    picked = None
    if not st.session_state.dialog:
        choice = st.pills("Try an example", list(SUGGESTIONS), label_visibility="collapsed")
        picked = SUGGESTIONS.get(choice)
    prompt = st.chat_input("Type in English or Hinglish, e.g. “yaar bahut tension ho rahi hai…”", submit_mode="disable")
    st.caption(":material/health_and_safety: Not a crisis or counselling service. In distress in India? "
               "Call Tele-MANAS **14416** (24×7).")

prompt = (prompt or picked or "").strip()
if prompt:
    counter = usage_counter()
    if counter["day"] != time.strftime("%Y-%m-%d"):
        counter.update(day=time.strftime("%Y-%m-%d"), n=0)
    st.session_state.dialog.append({"role": "user", "content": prompt})
    with box:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(esc(prompt))
        with st.chat_message("assistant", avatar=":material/favorite:"):
            if counter["n"] >= MAX_TURNS_PER_DAY:
                rec, reply = None, "The demo reached today's free-tier limit; please try again tomorrow."
            else:
                counter["n"] += 1
                with st.status(":shimmer[8 agents are reading, debating and voting…]", type="compact") as status:
                    t0 = time.time()
                    try:
                        rec = respond(st.session_state.dialog, early_rule)
                        reply = rec["response"]
                        status.update(label=f"Answered in {time.time() - t0:.0f} s · {rec['n_calls']} LLM calls",
                                      state="complete")
                    except Exception as e:  # keep the session usable
                        rec, reply = None, "Sorry, the model is busy right now. Please send your message again."
                        status.update(label=f"Model busy ({type(e).__name__})", state="error")
            st.markdown(esc(reply))
            if rec and rec.get("pred_strategy") in STRATEGY_PLAIN:
                st.caption(f"{STRATEGY_ICON[rec['pred_strategy']]} {rec['pred_strategy']}")
    st.session_state.dialog.append({"role": "assistant", "content": reply})
    st.session_state.traces.append(rec)

with info_col:
    st.markdown("#### What CodeMixESC understood")
    last = next((r for r in reversed(st.session_state.traces) if r), None)
    if last:
        insight_panel(last, load_pipeline()[1])
    else:
        empty_panel()
