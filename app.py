"""SpamSieve: a Streamlit app that flags spam text messages."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = Path("models/spamsieve.joblib")
METRICS_FILE = Path("models/metrics.json")

st.set_page_config(page_title="SpamSieve", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def spam_probability(model, messages):
    return model.predict_proba(pd.Series(messages, dtype=str))[:, 1]


if not MODEL_FILE.exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

model = load_model()

st.title("🛡️ SpamSieve")
st.caption("Paste a text message and find out whether it looks like spam.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Spam threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Messages with a spam probability at or above this are flagged. "
                               "Raise it to flag fewer messages, lower it to catch more.")
    if METRICS_FILE.exists():
        metrics = json.loads(METRICS_FILE.read_text())
        st.header("Model")
        st.write(f"`{metrics['model']}` trained on {metrics['n_messages']:,} messages")
        test = metrics["test"]
        st.write(f"Test precision **{test['precision']:.1%}** · recall **{test['recall']:.1%}**")

single, batch = st.tabs(["Check a message", "Check a CSV file"])

with single:
    message = st.text_area("Message", height=140,
                           placeholder="e.g. Congratulations! You've won a £500 voucher. Reply WIN to claim.")
    if st.button("Check", type="primary", disabled=not message.strip()):
        p = float(spam_probability(model, [message])[0])
        if p >= threshold:
            st.error(f"**Spam** — {p:.0%} spam probability")
        else:
            st.success(f"**Looks legitimate** — {p:.0%} spam probability")
        st.progress(p)

with batch:
    upload = st.file_uploader("Upload a CSV with one message per row", type="csv")
    if upload is not None:
        df = pd.read_csv(upload, encoding_errors="replace")
        column = st.selectbox("Which column holds the messages?", df.columns)
        if st.button("Check all", type="primary"):
            probs = spam_probability(model, df[column].fillna(""))
            df["spam_probability"] = probs.round(3)
            df["is_spam"] = probs >= threshold
            st.metric("Flagged as spam", f"{int(df['is_spam'].sum())} of {len(df)}")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download results", df.to_csv(index=False), "spamsieve_results.csv", "text/csv")
