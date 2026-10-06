import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from features import extract_features

BASE = Path(__file__).parent
COLOR = {"angry": "#dc2626", "happy": "#d97706", "neutral": "#475569", "sad": "#2563eb"}
GRAY = "#8b8b99"

st.set_page_config(page_title="Speech Emotion Detection", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1080px;}
.header {padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem; color: #fff;
         background: linear-gradient(120deg, #0f172a 0%, #1e1b4b 60%, #312e81 100%);
         border-bottom: 4px solid #6366f1;}
.header h1 {margin: 0; padding: 0; font-size: 1.9rem; color: #fff; letter-spacing: .2px;}
.header p {margin: .35rem 0 0; opacity: .85; font-size: 1rem;}
.pill {display: inline-block; margin: .8rem .35rem 0 0; padding: .2rem .8rem; border-radius: 999px;
       font-size: .8rem; font-weight: 600; color: #fff;}
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1.1rem 1.3rem;
       background: rgba(128,128,128,.05); margin-bottom: .8rem;}
.card h4 {margin: 0 0 .5rem; font-size: 1rem;}
.step {font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #6366f1; margin-bottom: .2rem;}
.result {border-radius: 16px; padding: 1.5rem 1.4rem; color: #fff;}
.result .k {font-size: .78rem; letter-spacing: .1em; text-transform: uppercase; opacity: .85;}
.result .label {font-size: 2.4rem; font-weight: 800; line-height: 1.15; margin: .2rem 0 .8rem;}
.result .meter {height: 8px; border-radius: 999px; background: rgba(255,255,255,.3); overflow: hidden;}
.result .meter div {height: 100%; background: #fff; border-radius: 999px;}
.result .conf {margin-top: .5rem; font-size: .95rem; opacity: .95;}
.bar-row {display: flex; align-items: center; margin: .6rem 0; gap: .8rem;}
.bar-name {width: 90px; font-weight: 600;}
.bar-track {flex: 1; height: 14px; border-radius: 999px; background: rgba(128,128,128,.2); overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.bar-val {width: 48px; text-align: right; font-variant-numeric: tabular-nums;}
.stat {padding: 1.1rem 1rem; border-radius: 14px; color: #fff;}
.stat .v {font-size: 1.9rem; font-weight: 800; line-height: 1.1;}
.stat .k {font-size: .85rem; opacity: .9; margin-top: .2rem;}
.tip {border-left: 4px solid #6366f1; padding: .65rem 1rem; border-radius: 8px;
      background: rgba(99,102,241,.08); margin-bottom: 1rem; font-size: .93rem;}
.stTabs [data-baseweb="tab"] {font-weight: 600; font-size: 1rem; padding-left: 1.1rem; padding-right: 1.1rem;}
</style>
""",
    unsafe_allow_html=True,
)

for key, default in [("history", []), ("seen", set()), ("rec_n", 0), ("up_n", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource
def load_model():
    return joblib.load(BASE / "model.joblib")


@st.cache_data
def load_metrics():
    with open(BASE / "metrics.json") as f:
        return json.load(f)


def predict_proba(file_bytes, model, feature_idx):
    vec = extract_features(io.BytesIO(file_bytes))[feature_idx]
    return pd.Series(model.predict_proba(vec.reshape(1, -1))[0], index=model.classes_)


def bars_html(values, classes, maximum=1.0, as_percent=True, highlight=None):
    rows = []
    for c in classes:
        v = float(values[c])
        width = 100 * v / maximum if maximum else 0
        label = f"{v:.0%}" if as_percent else f"{int(v)}"
        opacity = 1 if (highlight is None or c == highlight) else 0.4
        rows.append(
            f'<div class="bar-row"><div class="bar-name">{c.capitalize()}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{COLOR[c]};opacity:{opacity}"></div></div>'
            f'<div class="bar-val">{label}</div></div>'
        )
    return "".join(rows)


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    ax.title.set_color(GRAY)


def plot_waveform(file_bytes, sr, color):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr)
    fig, ax = plt.subplots(figsize=(9, 2.5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    librosa.display.waveshow(y, sr=sr, ax=ax, color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    style_axes(ax)
    fig.tight_layout()
    return fig, len(y) / sr


def plot_confusion(cm, classes):
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_alpha(0)
    names = [c.capitalize() for c in classes]
    for ax, data, title, fmt in [(axes[0], cm, "Counts", "d"), (axes[1], norm, "Row-normalised", ".2f")]:
        ax.patch.set_alpha(0)
        ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(classes)), names)
        ax.set_yticks(range(len(classes)), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        style_axes(ax)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "#111", fontweight="bold")
    fig.tight_layout()
    return fig


def add_to_history(file_bytes, source, proba):
    digest = hashlib.md5(file_bytes).hexdigest()
    if digest in st.session_state.seen:
        return
    st.session_state.seen.add(digest)
    row = {"Time": datetime.now().strftime("%H:%M:%S"), "Source": source,
           "Emotion": proba.idxmax().capitalize(), "Confidence": f"{proba.max():.0%}"}
    for c in proba.index:
        row[c.capitalize()] = f"{proba[c]:.0%}"
    st.session_state.history.insert(0, row)


def show_analysis(file_bytes, source, model, metrics):
    try:
        with st.spinner("Analysing audio..."):
            proba = predict_proba(file_bytes, model, metrics["feature_indices"])
            top = proba.idxmax()
            fig, duration = plot_waveform(file_bytes, metrics["sample_rate"], COLOR[top])
    except Exception as e:
        st.error(f"Could not process this audio. Please try another recording. ({e})")
        return
    add_to_history(file_bytes, source, proba)

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown(
            f'<div class="result" style="background:linear-gradient(135deg,{COLOR[top]},{COLOR[top]}d9)">'
            f'<div class="k">Detected emotion</div><div class="label">{top.capitalize()}</div>'
            f'<div class="meter"><div style="width:{proba.max()*100:.0f}%"></div></div>'
            f'<div class="conf">Confidence {proba.max():.0%}</div></div>',
            unsafe_allow_html=True,
        )
        if proba.max() < 0.5:
            st.warning("Low confidence: the model is unsure about this clip.")
        if duration > 10:
            st.warning("This clip is long. The model was trained on clips of about 3-4 seconds.")
        if duration < 1:
            st.warning("This clip is very short. Try speaking for 3-5 seconds.")
    with right:
        st.markdown(
            f'<div class="card"><h4>Probability per emotion</h4>{bars_html(proba, metrics["classes"], highlight=top)}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("**Waveform**")
    st.pyplot(fig)
    st.caption(f"Duration after trimming silence: {duration:.1f} s. Saved to the History tab.")


def new_recording():
    st.session_state.rec_n += 1


def new_upload():
    st.session_state.up_n += 1


model = load_model()
metrics = load_metrics()
classes = metrics["classes"]

pills = "".join(f'<span class="pill" style="background:{COLOR[c]}">{c.capitalize()}</span>' for c in classes)
st.markdown(
    f'<div class="header"><h1>Speech Emotion Detection</h1>'
    f'<p>Record your voice or upload a clip, and the model predicts the emotion in the speech.</p>{pills}</div>',
    unsafe_allow_html=True,
)

tab_predict, tab_history, tab_dash, tab_about = st.tabs(["Predict", "History", "Dashboard", "About"])

with tab_predict:
    st.markdown('<div class="step">Step 1</div>**Choose how to give the audio**', unsafe_allow_html=True)
    mode = st.radio("Input method", ["Record my voice", "Upload an audio file"],
                    horizontal=True, label_visibility="collapsed", key="mode")
    st.markdown(
        '<div class="tip"><b>Tip:</b> the model was trained on acted speech. Speak clearly for 3-5 seconds in a quiet room '
        'and exaggerate the emotion, for example say <i>"Kids are talking by the door"</i> in an angry, happy or sad voice.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="step">Step 2</div>**Provide the audio**', unsafe_allow_html=True)

    if mode == "Record my voice":
        recording = st.audio_input("Press the microphone, speak, then press stop", key=f"rec_{st.session_state.rec_n}")
        if recording is None:
            st.info("Allow microphone access in your browser, then press the microphone button and speak.")
        else:
            st.button("Record a new clip", on_click=new_recording, key="again_rec")
            show_analysis(recording.getvalue(), "Microphone", model, metrics)
    else:
        uploaded = st.file_uploader("Upload a WAV file", type=["wav"], key=f"up_{st.session_state.up_n}")
        if uploaded is None:
            st.info("Choose a WAV file to see the predicted emotion, probabilities and waveform.")
        else:
            st.button("Upload another file", on_click=new_upload, key="again_up")
            st.audio(uploaded.getvalue(), format="audio/wav")
            show_analysis(uploaded.getvalue(), f"Upload: {uploaded.name}", model, metrics)

with tab_history:
    hist = st.session_state.history
    st.markdown("#### Prediction history")
    if not hist:
        st.info("No predictions yet. Record or upload audio in the Predict tab and it will appear here.")
    else:
        hdf = pd.DataFrame(hist)
        counts = hdf["Emotion"].str.lower().value_counts().reindex(classes).fillna(0)
        c1, c2 = st.columns([1, 1.4], gap="large")
        with c1:
            st.markdown(
                f'<div class="stat" style="background:#4338ca"><div class="v">{len(hdf)}</div>'
                f'<div class="k">Predictions in this session</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="card"><h4>Predicted emotions</h4>'
                f'{bars_html(counts, classes, maximum=max(counts.max(), 1), as_percent=False)}</div>',
                unsafe_allow_html=True,
            )
        st.dataframe(hdf, hide_index=True)
        b1, b2, _ = st.columns([1, 1, 3])
        b1.download_button("Download CSV", hdf.to_csv(index=False), "prediction_history.csv", "text/csv")
        if b2.button("Clear history"):
            st.session_state.history = []
            st.session_state.seen = set()
            st.rerun()
        st.caption("History is kept for the current browser session only.")

with tab_dash:
    st.markdown("#### Model performance on unseen actors")
    c1, c2, c3 = st.columns(3)
    for col, value, key, color in [
        (c1, f"{metrics['accuracy']:.1%}", "Accuracy", "#4338ca"),
        (c2, f"{metrics['macro_f1']:.3f}", "Macro F1", "#7c3aed"),
        (c3, f"{metrics['n_test']}", "Test clips", "#0f766e"),
    ]:
        col.markdown(
            f'<div class="stat" style="background:{color}"><div class="v">{value}</div><div class="k">{key}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        f"Model: {metrics['model']} | Features: {metrics['feature_set']} | Training: {metrics['training_regime']} | "
        f"Train actors {min(metrics['train_actors'])}-{max(metrics['train_actors'])}, "
        f"test actors {min(metrics['test_actors'])}-{max(metrics['test_actors'])}"
    )

    left, right = st.columns(2, gap="large")
    with left:
        cc = pd.Series(metrics["class_counts_all"])
        st.markdown(
            f'<div class="card"><h4>Class counts (672 clips)</h4>'
            f'{bars_html(cc, classes, maximum=cc.max(), as_percent=False)}</div>',
            unsafe_allow_html=True,
        )
    with right:
        report = metrics["per_class_report"]
        table = pd.DataFrame(
            {c.capitalize(): {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in classes}
        ).T
        st.markdown("**Per-class scores**")
        st.dataframe(table.round(3))

    st.markdown("**Confusion matrix**")
    st.pyplot(plot_confusion(metrics["confusion_matrix"], classes))

    with st.expander("All experiments (cross-validation vs test)"):
        st.dataframe(pd.DataFrame(metrics["experiments"]).round(3))

with tab_about:
    st.markdown(
        """
#### About this project

**Data:** RAVDESS speech audio, 4 emotions (angry, happy, sad, neutral), 672 clips from 24 actors.

**Features:** MFCC, delta-MFCC, chroma, log-mel, spectral contrast and loudness/brightness statistics per clip.

**Model:** scikit-learn classifier with balanced class weights, because the neutral class is half the size of the others.

**Evaluation:** actors are split, so the test voices were never seen during training. This avoids data leakage.

**How to use:** open the Predict tab, record your voice or upload a file, and read the detected emotion and the probability for each class. Every prediction is saved in the History tab.

**Limitations:** the clips are acted speech recorded in a studio by North American actors. Accuracy on real phone calls,
noisy audio, other accents or other languages will be lower.
"""
    )
