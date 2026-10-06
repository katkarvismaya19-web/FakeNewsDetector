"""Streamlit UI.  Run: streamlit run app.py"""
from datetime import datetime

import pandas as pd
import streamlit as st

import auth
import ui
from config import ENABLE_LLM
from data import detect_lang

st.set_page_config(page_title="Fake News Detector", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)
html = lambda s: st.markdown(s, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (first run downloads them, about a minute)…")
def load(with_llm):
    from predict import Detector
    return Detector(("rnn", "lstm", "bert", "llm") if with_llm else ("rnn", "lstm", "bert"))


def login_page():
    left, _, right = st.columns([1.1, 0.15, 0.9], vertical_alignment="center")
    with left:
        html(ui.brand())
        html('<div class="pitch"><h1>Check a news story before you share it</h1>'
             '<p>Paste any Hindi or English headline, article or forwarded message. '
             'You get a verdict, a confidence score and the fact-checks behind it.</p><ul>'
             '<li>Four models compared: BERT, LSTM, RNN and an LLM</li>'
             '<li>Cross-checked against published fact-checks and news coverage</li>'
             '<li>Calibrated confidence, so 90% means 90%</li></ul></div>')
    with right:
        with st.container(border=True):
            tab_in, tab_up = st.tabs(["Log in", "Create account"])
            with tab_in:
                with st.form("login", border=False):
                    email = st.text_input("Email", placeholder="name@college.edu")
                    pw = st.text_input("Password", type="password")
                    if st.form_submit_button("Log in", type="primary", use_container_width=True):
                        name = auth.login(email, pw)
                        if name:
                            st.session_state.user = name
                            st.rerun()
                        st.error("Incorrect email or password.")
            with tab_up:
                with st.form("register", border=False):
                    name = st.text_input("Full name", placeholder="Priya Sharma")
                    email = st.text_input("Email ", placeholder="name@college.edu")
                    pw = st.text_input("Password", type="password", help="At least 8 characters")
                    if st.form_submit_button("Create account", type="primary", use_container_width=True):
                        ok, msg = auth.register(name, email, pw)
                        (st.success if ok else st.error)(msg)


def header():
    c1, c2 = st.columns([5, 1], vertical_alignment="center")
    with c1:
        html(ui.brand())
    with c2:
        with st.popover(st.session_state.user, use_container_width=True):
            if st.button("Log out", use_container_width=True):
                st.session_state.clear()
                st.rerun()
    html('<div class="hr"></div>')


def run_check(text, with_llm):
    from evidence import combine, verify
    lang = detect_lang(text)
    with st.status("Checking…", expanded=False) as s:
        s.update(label="Running models…")
        probs = load(with_llm).predict_all(text)
        s.update(label="Searching fact-checks and news…")
        ev = verify(text)
        s.update(label="Done", state="complete")
    final = combine(probs["bert"], ev)
    st.session_state.result = dict(text=text, lang=lang, probs=probs, ev=ev, final=final)
    st.session_state.setdefault("history", []).insert(0, {
        "Time": datetime.now().strftime("%H:%M"), "Claim": text[:90] + ("…" if len(text) > 90 else ""),
        "Language": "Hindi" if lang == "hi" else "English",
        "Verdict": "Fake" if final >= 0.5 else "Real", "Confidence": f"{max(final, 1 - final) * 100:.0f}%"})


def results(r):
    ev = r["ev"]
    top_fc = next((s for s in ev["sources"] if s["type"] == "fact-check"), None)
    basis = {"fact-check": f"confirmed by {top_fc['publisher'] if top_fc else 'a published fact-check'}",
             "news-debunk": "news reports call this claim false",
             "news-corroborated": "reported by multiple news outlets"}.get(ev["strength"],
                                                                         "model only, no matching source found")
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
    with c1:
        html(ui.verdict_card(r["final"], basis))
    with c2:
        html(ui.stat("Language", "Hindi" if r["lang"] == "hi" else "English"))
    with c3:
        html(ui.stat("Sources matched", str(len(ev["sources"]))))
    with c4:
        html(ui.stat("Verification", {"fact-check": "Fact-checked", "news-debunk": "News",
                                      "news-corroborated": "News"}.get(ev["strength"], "Unverified")))
    st.write("")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Model comparison**")
            html(ui.model_table(r["probs"]))
            st.caption("Final verdict uses BERT combined with source evidence.")
    with right:
        with st.container(border=True):
            st.markdown("**Sources**")
            if ev["sources"]:
                html("".join(ui.source_card(s) for s in ev["sources"][:6]))
            else:
                html(ui.empty("No fact-check or news report matched this claim."))


def dashboard():
    header()
    with st.container(border=True):
        text = st.text_area("News article or claim", height=140, label_visibility="collapsed",
                            placeholder="Paste a headline, article or forwarded message in Hindi or English")
        a, b = st.columns([3, 1], vertical_alignment="center")
        with_llm = a.toggle("Include LLM in comparison (slower)") if ENABLE_LLM else False
        if b.button("Check news", type="primary", use_container_width=True):
            if len(text.strip()) < 15:
                st.warning("Enter at least one full sentence to check.")
            else:
                run_check(text.strip(), with_llm)
    st.write("")
    if "result" in st.session_state:
        results(st.session_state.result)
    else:
        html(ui.empty("Results appear here: verdict, confidence, model comparison and sources."))
    if st.session_state.get("history"):
        st.write("")
        st.markdown("**Recent checks**")
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)


if "user" in st.session_state:
    dashboard()
else:
    login_page()
