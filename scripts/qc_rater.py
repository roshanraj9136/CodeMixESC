"""Manual quality check of ESConv-HiEn, one conversation at a time (English and Hinglish side by
side, two 1-5 ratings). Reads the sample of scripts/quality_check.py and writes the ratings into
the same sheets, so `quality_check.py report` works unchanged.

    python scripts/quality_check.py sample        # once: picks the 20% sample, writes the sheets
    streamlit run scripts/qc_rater.py
"""
import csv
import json
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from codemixesc.esconv import hien_dir  # noqa: E402

QC = os.path.join(hien_dir(), "qc")
HEADER = ["level", "conv_id", "n_turns", "naturalness (1-5)", "meaning (1-5)", "comments"]
SCALE = {1: "1 · poor", 2: "2", 3: "3 · okay", 4: "4", 5: "5 · excellent"}

st.set_page_config(page_title="ESConv-HiEn quality check", page_icon=":material/fact_check:", layout="wide")


@st.cache_data
def load_conversations(level):
    with open(os.path.join(hien_dir(), f"test_{level}.json"), encoding="utf-8") as f:
        return json.load(f)


def read_sheet(level):
    with open(os.path.join(QC, f"sheet_{level}.csv"), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_sheet(level, rows):
    path = os.path.join(QC, f"sheet_{level}.csv")
    with open(path + ".tmp", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)


if not os.path.exists(os.path.join(QC, "sample.json")):
    st.error("Run `python scripts/quality_check.py sample` first.", icon=":material/error:")
    st.stop()

st.title("ESConv-HiEn quality check")
level = st.segmented_control("Mixing level", ["light", "heavy"], default="light", format_func=str.title,
                             required=True)
rows = read_sheet(level)
done = sum(1 for r in rows if r["naturalness (1-5)"] and r["meaning (1-5)"])
st.progress(done / max(1, len(rows)), text=f"{done} of {len(rows)} conversations rated ({level.title()})")

ids = [int(r["conv_id"]) for r in rows]
first_open = next((k for k, r in enumerate(rows) if not r["naturalness (1-5)"]), 0)
key = f"pos_{level}"
if key not in st.session_state:
    st.session_state[key] = first_open
pos = st.session_state[key]
row = rows[pos]
conv = load_conversations(level)[ids[pos]]

with st.container(horizontal=True, vertical_alignment="center"):
    if st.button("Previous", icon=":material/arrow_back:", disabled=pos == 0):
        st.session_state[key] = pos - 1
        st.rerun()
    st.markdown(f"**Conversation {ids[pos]}** · {len(conv['dialog'])} turns · {pos + 1}/{len(rows)}")
    if st.button("Next", icon=":material/arrow_forward:", disabled=pos == len(rows) - 1):
        st.session_state[key] = pos + 1
        st.rerun()

left, right = st.columns([3, 1], gap="large")
with left:
    with st.container(height=560, border=True):
        head = st.columns([1, 5, 5])
        head[1].markdown("**English original**")
        head[2].markdown("**Hinglish rewrite**")
        for t in conv["dialog"]:
            c = st.columns([1, 5, 5])
            who = "Seeker" if t["speaker"] == "seeker" else "Supporter"
            c[0].badge(who, color="blue" if who == "Seeker" else "violet")
            strat = (t.get("annotation") or {}).get("strategy") if t["speaker"] == "supporter" else None
            c[1].markdown(t.get("content_en", "") + (f"  \n:gray[*{strat}*]" if strat else ""))
            c[2].markdown(t["content"] + (f"  \n:gray[CMI {t['cmi']:.2f}]" if t.get("band_checked") else ""))
with right:
    with st.form(f"rate_{level}_{ids[pos]}", border=True):
        st.markdown("**Naturalness**  \nDoes the Hinglish read like real chat Hinglish?")
        nat = st.radio("Naturalness", list(SCALE), format_func=SCALE.get, horizontal=False,
                       index=int(row["naturalness (1-5)"]) - 1 if row["naturalness (1-5)"] else None,
                       label_visibility="collapsed")
        st.markdown("**Meaning**  \nSame meaning, emotion and strategy as the English?")
        mean = st.radio("Meaning", list(SCALE), format_func=SCALE.get,
                        index=int(row["meaning (1-5)"]) - 1 if row["meaning (1-5)"] else None,
                        label_visibility="collapsed")
        comment = st.text_area("Comments (optional)", value=row["comments"], height=90)
        if st.form_submit_button("Save and next", icon=":material/save:", type="primary", width="stretch"):
            if nat is None or mean is None:
                st.warning("Pick both ratings.")
            else:
                row.update({"naturalness (1-5)": nat, "meaning (1-5)": mean, "comments": comment})
                write_sheet(level, rows)
                st.session_state[key] = min(pos + 1, len(rows) - 1)
                st.rerun()
    st.caption("Ratings are saved to `data/esconv_hien/qc/sheet_{level}.csv`; "
               "`python scripts/quality_check.py report` computes the agreement with the LLM rater.")
