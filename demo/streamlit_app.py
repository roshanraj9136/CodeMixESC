"""CodeMixESC live demo (Streamlit).

Type as a help-seeker in English or Roman-script Hinglish. The full CodeMixESC multi-agent
pipeline answers, and each reply can be expanded to see what every stage did: the register
profile R, the dialogue analysis, the retrieved cases, the strategies, the vote and the
Register Gate.

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
DELTA = 0.2
MAX_TURNS_PER_DAY = 400

st.set_page_config(page_title="CodeMixESC demo", page_icon=":material/forum:", layout="wide")


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


@st.cache_resource(show_spinner="Loading the language identifier, the retriever and the case bank (first visit only)…")
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


def esc(t):
    return html.escape(str(t)).replace("\n", " ").replace("$", "\\$")


def render_trace(rec, retriever):
    """One timeline step per pipeline stage of a CodeMixESC record."""
    from codemixesc.profiler import describe_register, is_plain_english
    R = rec.get("R")
    if R:
        mix, script = describe_register(R)
        with st.status("Code-Mix Profiler: register profile R", type="step", state="complete"):
            st.markdown(f"{esc(mix)}, {esc(script)} · CMI **{R['cmi']:.2f}** · Hindi share "
                        f"**{100 * R['hi_frac']:.0f}%** over {R['n_lang']} words"
                        + (" · treated as **plain English**" if is_plain_english(R) else ""))
    if rec.get("path") == "single":
        with st.status("Single zero-shot agent (with the register instruction)", type="step", state="complete"):
            st.markdown("Early turn, or the decision maker found the dialogue not yet complex "
                        "(as in MultiAgentESC).")
    else:
        a = rec.get("analysis") or {}
        with st.status("Stage 1 · emotion, cause and intention agents", type="step", state="complete"):
            st.markdown(f"**Emotion:** {esc(a.get('emotion'))}  \n**Cause:** {esc(a.get('cause'))}  \n"
                        f"**Intention:** {esc(a.get('intention'))}")
        ids = rec.get("retrieved") or []
        with st.status("Stage 2 · cross-lingual retrieval (top 3 of 10 English cases)", type="step", state="complete"):
            for i in ids[:3]:
                b = retriever.bank[i]
                st.markdown(f"- “{esc(b['post'][:150])}” → *[{esc(b['strategy'])}]* {esc(b['response'][:130])}")
        with st.status("Stage 2 · strategy deliberation (3 agents)", type="step", state="complete"):
            st.markdown(", ".join(f"`{s}`" for s in rec.get("strategies") or []) or "—")
        with st.status("Stage 3 · one reply per strategy, debate → reflection → vote", type="step", state="complete"):
            for s, r in rec.get("candidates") or []:
                st.markdown(f"- *[{esc(s)}]* {esc(r)}")
            v = rec.get("vote") or {}
            if v.get("strategies"):
                st.markdown(f"**Winner:** {esc(', '.join(v['strategies']))}"
                            + (" (tie broken by the judge)" if rec.get("judge") else ""))
    g = rec.get("gate") or {}
    if g:
        fired = g.get("triggered")
        label = "Register Gate: " + ("fired" + (", rewrite kept" if g.get("accepted") else ", original kept")
                                     if fired else "passed")
        with st.status(label, type="step", state="complete"):
            st.markdown(f"Reply {100 * g['hi_frac_before']:.0f}% Hindi vs the user's target "
                        f"{100 * g['hi_frac_target']:.0f}% (δ = {g['delta']})"
                        + (f" → after the rewrite {100 * g['hi_frac_after']:.0f}%" if g.get("accepted") else ""))


def respond(dialog, early_rule):
    from codemixesc.esconv import json2natural
    count = len(dialog) + 1  # dialog index after this supporter turn, as in the base main.py
    sample = {"uid": f"demo-{len(dialog)}", "conv_id": -1, "turn": len(dialog), "strategy": "", "reference": "",
              "context_msgs": list(dialog), "context": json2natural(dialog), "post": dialog[-1]["content"],
              "early": early_rule and count <= 5}
    system, _ = load_pipeline()
    return system.respond(sample, "demo")


# ------------------------------------------------------------------ page
with st.sidebar:
    st.header("CodeMixESC")
    st.markdown("Emotional support for **code-mixed Hinglish** help-seekers: MultiAgentESC (EMNLP 2025) + "
                "a Code-Mix Profiler, a cross-lingual retriever, register-aware agents and a Register Gate.")
    st.markdown("[:material/code: Code and results on GitHub](https://github.com/roshanraj9136/CodeMixESC)  \n"
                f"[:material/neurology: Fine-tuned retriever](https://huggingface.co/{RETRIEVER_REPO})")
    early_rule = st.toggle("Base-paper rule: first turns use one agent", value=False,
                           help="MultiAgentESC answers the first turns of a dialogue with a single zero-shot "
                                "agent. Off = the full pipeline from the first message.")
    if st.button("New conversation", icon=":material/refresh:", width="stretch"):
        st.session_state.dialog, st.session_state.traces = [], []
        st.rerun()
    st.caption(f"Agents: `{MODEL}` (Gemini API free tier). Course project of Roshan Raj, IIT Bhilai (12341830).")
    st.warning("Research prototype, not a crisis or counselling service. In distress? Tele-MANAS **14416** "
               "(India, 24×7).", icon=":material/health_and_safety:")

st.title("Talk to CodeMixESC")
st.caption("Write as someone looking for emotional support, in English or Hinglish. Each reply comes from "
           "8 cooperating LLM agents (about 20–60 s); open *pipeline trace* to see what they did.")

if "dialog" not in st.session_state:
    st.session_state.dialog, st.session_state.traces = [], []

key = api_key()
if not key:
    st.error("GEMINI_API_KEY is not configured for this app.", icon=":material/key_off:")
    st.stop()
os.environ["GEMINI_API_KEY"] = key

for i, msg in enumerate(st.session_state.dialog):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and st.session_state.traces[i // 2] is not None:
            rec = st.session_state.traces[i // 2]
            with st.expander(f"Pipeline trace · {rec['n_calls']} LLM calls, {rec['latency']:.0f} s", type="compact"):
                render_trace(rec, load_pipeline()[1])
        st.markdown(esc(msg["content"]))

SUGGESTIONS = {
    ":orange[:material/school:] Exam result stress": "yaar exams ka result aaya hai aur bahut tension ho rahi hai, "
                                                     "ghar pe kaise bataun samajh nahi aa raha",
    ":blue[:material/work:] No job yet": "Mujhe lagta hai main kabhi job nahi dhoond paunga, sab friends settle ho gaye "
                                         "aur main abhi bhi wahi hoon",
    ":green[:material/group:] Friend stopped talking": "meri best friend ne mujhse baat karna band kar diya, "
                                                       "I don't even know what I did wrong",
    ":violet[:material/language:] English": "I lost my job last month and honestly I feel useless",
}
picked = None
if not st.session_state.dialog:
    choice = st.pills("Try an example", list(SUGGESTIONS), label_visibility="collapsed")
    picked = SUGGESTIONS.get(choice)

prompt = st.chat_input("Type in English or Hinglish, e.g. 'yaar bahut tension ho rahi hai…'", submit_mode="disable")
prompt = prompt or picked
if prompt:
    counter = usage_counter()
    if counter["day"] != time.strftime("%Y-%m-%d"):
        counter.update(day=time.strftime("%Y-%m-%d"), n=0)
    st.session_state.dialog.append({"role": "user", "content": prompt.strip()})
    with st.chat_message("user"):
        st.markdown(esc(prompt.strip()))
    with st.chat_message("assistant"):
        if counter["n"] >= MAX_TURNS_PER_DAY:
            reply, rec = "The demo reached today's free-tier limit; please try again tomorrow.", None
            st.markdown(reply)
        else:
            counter["n"] += 1
            with st.status(":shimmer[8 agents are thinking]", type="compact") as status:
                t0 = time.time()
                try:
                    rec = respond(st.session_state.dialog, early_rule)
                    reply = rec["response"]
                    render_trace(rec, load_pipeline()[1])
                    status.update(label=f"Pipeline trace · {rec['n_calls']} LLM calls, {time.time() - t0:.0f} s",
                                  state="complete")
                except Exception as e:  # keep the session usable
                    rec, reply = None, "Sorry, the model is busy right now. Please send your message again."
                    status.update(label=f"Error: {type(e).__name__}", state="error")
            st.markdown(esc(reply))
    st.session_state.dialog.append({"role": "assistant", "content": reply})
    st.session_state.traces.append(rec)
