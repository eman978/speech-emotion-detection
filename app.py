import io
import json
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
EMOJI = {"angry": "😠", "happy": "😄", "neutral": "😐", "sad": "😢"}
COLOR = {"angry": "#ef4444", "happy": "#f59e0b", "neutral": "#64748b", "sad": "#3b82f6"}
GRAY = "#8b8b99"

st.set_page_config(page_title="Speech Emotion Detection", page_icon="🎙️", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 1.8rem 2rem; border-radius: 20px; margin-bottom: 1.4rem; color: #fff;
       background: linear-gradient(135deg, #4f46e5 0%, #9333ea 55%, #ec4899 100%);
       box-shadow: 0 10px 30px rgba(79, 70, 229, .25);}
.hero h1 {margin: 0; padding: 0; font-size: 2.1rem; color: #fff;}
.hero p {margin: .4rem 0 0; font-size: 1.05rem; opacity: .92;}
.chips span {display: inline-block; margin: .7rem .4rem 0 0; padding: .25rem .75rem; border-radius: 999px;
             background: rgba(255,255,255,.18); font-size: .85rem;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 18px; padding: 1.2rem 1.4rem;
       background: rgba(128,128,128,.06);}
.result {border-radius: 20px; padding: 1.4rem 1.6rem; color: #fff; text-align: center;}
.result .emoji {font-size: 3.4rem; line-height: 1.1;}
.result .label {font-size: 2rem; font-weight: 800; margin-top: .2rem;}
.result .conf {font-size: 1rem; opacity: .92;}
.bar-row {display: flex; align-items: center; margin: .55rem 0; gap: .7rem;}
.bar-name {width: 105px; font-weight: 600;}
.bar-track {flex: 1; height: 16px; border-radius: 999px; background: rgba(128,128,128,.2); overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.bar-val {width: 52px; text-align: right; font-variant-numeric: tabular-nums;}
.stat {text-align: center; padding: 1.1rem .6rem; border-radius: 18px; color: #fff;}
.stat .v {font-size: 2rem; font-weight: 800;}
.stat .k {font-size: .9rem; opacity: .9;}
.tip {border-left: 4px solid #9333ea; padding: .6rem 1rem; border-radius: 8px; background: rgba(147,51,234,.08);
      margin-bottom: 1rem; font-size: .95rem;}
</style>
""",
    unsafe_allow_html=True,
)


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
        opacity = 1 if (highlight is None or c == highlight) else 0.45
        rows.append(
            f'<div class="bar-row"><div class="bar-name">{EMOJI[c]} {c.capitalize()}</div>'
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
    fig, ax = plt.subplots(figsize=(9, 2.6))
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_alpha(0)
    for ax, data, title, fmt in [(axes[0], cm, "Counts", "d"), (axes[1], norm, "Row-normalised", ".2f")]:
        ax.patch.set_alpha(0)
        ax.imshow(data, cmap="Purples")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        style_axes(ax)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "#222", fontweight="bold")
    fig.tight_layout()
    return fig


def show_analysis(file_bytes, model, metrics):
    try:
        with st.spinner("Analysing your voice..."):
            proba = predict_proba(file_bytes, model, metrics["feature_indices"])
            top = proba.idxmax()
            fig, duration = plot_waveform(file_bytes, metrics["sample_rate"], COLOR[top])
    except Exception as e:
        st.error(f"Could not process this audio. Please try another recording. ({e})")
        return

    left, right = st.columns([1, 1.6], gap="large")
    with left:
        st.markdown(
            f'<div class="result" style="background:linear-gradient(135deg,{COLOR[top]},{COLOR[top]}cc)">'
            f'<div class="emoji">{EMOJI[top]}</div><div class="label">{top.capitalize()}</div>'
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
            f'<div class="card"><b>Probability per emotion</b>'
            f'{bars_html(proba, metrics["classes"], highlight=top)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("##### Waveform")
    st.pyplot(fig)
    st.caption(f"Duration after trimming silence: {duration:.1f} s")


model = load_model()
metrics = load_metrics()
classes = metrics["classes"]

st.markdown(
    """
<div class="hero">
  <h1>🎙️ Speech Emotion Detection</h1>
  <p>Speak or upload a short clip and the AI tells you the emotion in your voice.</p>
  <div class="chips"><span>😠 Angry</span><span>😄 Happy</span><span>😐 Neutral</span><span>😢 Sad</span></div>
</div>
""",
    unsafe_allow_html=True,
)

tab_predict, tab_results, tab_about = st.tabs(["🎯 Predict", "📊 Results", "ℹ️ About"])

with tab_predict:
    st.markdown(
        '<div class="tip"><b>Tip:</b> the model was trained on acted speech. Speak clearly for 3-5 seconds in a quiet room '
        'and exaggerate the emotion, for example say <i>"Kids are talking by the door"</i> in an angry, happy or sad voice.</div>',
        unsafe_allow_html=True,
    )
    rec_tab, up_tab = st.tabs(["🎤 Record your voice", "📁 Upload a WAV file"])

    with rec_tab:
        recording = st.audio_input("Click the microphone, speak, then click stop")
        if recording is None:
            st.info("Allow microphone access in your browser, then press the mic button and speak.")
        else:
            show_analysis(recording.getvalue(), model, metrics)

    with up_tab:
        uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
        if uploaded is None:
            st.info("Upload a WAV file to see the predicted emotion, probabilities and waveform.")
        else:
            st.audio(uploaded.getvalue(), format="audio/wav")
            show_analysis(uploaded.getvalue(), model, metrics)

with tab_results:
    c1, c2, c3 = st.columns(3)
    for col, value, key, grad in [
        (c1, f"{metrics['accuracy']:.1%}", "Accuracy", "#4f46e5,#6366f1"),
        (c2, f"{metrics['macro_f1']:.3f}", "Macro F1", "#9333ea,#a855f7"),
        (c3, f"{metrics['n_test']}", "Test clips (unseen actors)", "#ec4899,#f472b6"),
    ]:
        col.markdown(
            f'<div class="stat" style="background:linear-gradient(135deg,{grad})">'
            f'<div class="v">{value}</div><div class="k">{key}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        f"Model: {metrics['model']} | Features: {metrics['feature_set']} | Training: {metrics['training_regime']} | "
        f"Train actors {min(metrics['train_actors'])}-{max(metrics['train_actors'])}, "
        f"test actors {min(metrics['test_actors'])}-{max(metrics['test_actors'])}"
    )

    left, right = st.columns(2, gap="large")
    with left:
        counts = pd.Series(metrics["class_counts_all"])
        st.markdown(
            f'<div class="card"><b>Class counts (672 clips)</b>'
            f'{bars_html(counts, classes, maximum=counts.max(), as_percent=False)}</div>',
            unsafe_allow_html=True,
        )
    with right:
        report = metrics["per_class_report"]
        table = pd.DataFrame(
            {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in classes}
        ).T
        st.markdown("**Per-class scores**")
        st.dataframe(table.round(3))

    st.markdown("##### Confusion matrix")
    st.pyplot(plot_confusion(metrics["confusion_matrix"], classes))

    with st.expander("All experiments (cross-validation vs test)"):
        st.dataframe(pd.DataFrame(metrics["experiments"]).round(3))

with tab_about:
    st.markdown(
        """
**Data:** RAVDESS speech audio, 4 emotions (angry, happy, sad, neutral), 672 clips from 24 actors.

**Features:** MFCC, delta-MFCC, chroma, log-mel, spectral contrast and loudness/brightness statistics per clip.

**Model:** scikit-learn classifier with balanced class weights, because the neutral class is half the size of the others.

**Evaluation:** actors are split, so the test voices were never seen during training. This avoids data leakage.

**Limitations:** the clips are acted speech recorded in a studio by North American actors. Accuracy on real phone calls,
noisy audio, other accents or other languages will be lower.
"""
    )
