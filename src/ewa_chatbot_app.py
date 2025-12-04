# src/ewa_chatbot_app.py

import os
import pandas as pd
import streamlit as st
import win32com.client as win32

from src.config import HISTORY_CSV, DOCX_DIR
from src.ewa_processor import (
    build_history_from_folder,
    parse_single_report,
    compute_deviation,
    score_risk,
    encode_colors,
    load_model,
    predict_overall_from_sections,
)
from src.utils_openai import gpt_answer, gpt_risk_summary


def ensure_history():
    if os.path.exists(HISTORY_CSV):
        return pd.read_csv(HISTORY_CSV, parse_dates=["report_date"])
    return build_history_from_folder()


def convert_doc(uploaded):
    DOCX_DIR.mkdir(exist_ok=True, parents=True)
    fpath = DOCX_DIR / uploaded.name
    with open(fpath, "wb") as f:
        f.write(uploaded.read())

    if uploaded.name.lower().endswith(".docx"):
        return str(fpath)

    base = fpath.with_suffix(".docx")
    word = win32.Dispatch("Word.Application")
    doc = word.Documents.Open(str(fpath))
    doc.SaveAs(str(base), FileFormat=16)
    doc.Close()
    word.Quit()
    return str(base)


def fmt(v):
    return {"GREEN": "🟩 GREEN", "YELLOW": "🟨 YELLOW", "RED": "🟥 RED"}.get(v, "⬜ NA")


def build_context(hist, new, dev, lvl, score, pred):
    parts = [
        "History:",
        hist.to_string(index=False),
        "\nNew Report:",
        new.to_string(),
    ]
    if not dev.empty:
        parts.append("\nDeviations:")
        parts.append(dev.to_string(index=False))
    parts.append(f"\nRisk: {lvl} (score {score})")
    if pred:
        parts.append(f"Predicted: {pred}")
    return "\n\n".join(parts)


# UI
st.set_page_config(page_title="SAP EWA Analyzer", layout="wide")
st.title("📊 SAP EWA Analyzer + Chatbot")

hist = ensure_history()
hist_disp = hist.copy()
for col in hist_disp.columns:
    if col not in ("system", "report_date"):
        hist_disp[col] = hist_disp[col].apply(fmt)
st.dataframe(hist_disp, use_container_width=True)

encoded = encode_colors(hist)
kpis = [c for c in encoded.columns if c not in ("system", "report_date")]
metric = st.selectbox("Trend KPI", kpis)
chart = pd.DataFrame({"level": encoded[metric], "date": hist["report_date"]}).set_index("date")
st.line_chart(chart)

latest = hist.iloc[-1]
st.write("Latest:", latest["report_date"].date())

try:
    model, feats = load_model()
except:
    model = feats = None

uploaded = st.file_uploader("Upload new EWA (.doc/.docx)", type=["doc", "docx"])
if uploaded:
    docx = convert_doc(uploaded)
    rec = parse_single_report(docx)
    new = pd.Series(rec)

    new_disp = new.copy()
    for k in new_disp.index:
        if k not in ("system", "report_date"):
            new_disp[k] = fmt(new_disp[k])
    st.dataframe(new_disp.to_frame("Status"))

    dev = compute_deviation(hist, new)
    if not dev.empty:
        dev["previous"] = dev["previous"].apply(fmt)
        dev["new"] = dev["new"].apply(fmt)
        st.warning("Differences found:")
        st.dataframe(dev)

    lvl, score, info = score_risk(hist, new)
    st.info(f"Risk: {lvl} (Score {score})")
    if model:
        pred = predict_overall_from_sections(model, feats, new)
        st.info(f"ML Prediction: {pred}")
    else:
        pred = None

    ctx = build_context(hist, new, dev, lvl, score, pred)
    if st.button("AI Risk Summary"):
        st.write(gpt_risk_summary(ctx))

    q = st.text_input("Ask a question:")
    if st.button("Ask"):
        st.write(gpt_answer(q, ctx))
