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
EMOJI = {"angry": "😠", "happy": "😊", "neutral": "😐", "sad": "😢"}

st.set_page_config(page_title="Speech Emotion Detection", page_icon="🎙️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(BASE / "model.joblib")


@st.cache_data
def load_metrics():
    with open(BASE / "metrics.json") as f:
        return json.load(f)


def predict_proba(file_bytes, model, feature_idx):
    vec = extract_features(io.BytesIO(file_bytes))[feature_idx]
    proba = model.predict_proba(vec.reshape(1, -1))[0]
    return pd.Series(proba, index=model.classes_)


def plot_waveform(file_bytes, sr):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig, len(y) / sr


def plot_confusion(cm, classes):
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, title, fmt in [(axes[0], cm, "Counts", "d"), (axes[1], norm, "Row-normalised", ".2f")]:
        ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(len(classes)):
            for j in range(len(classes)):
                color = "white" if data[i, j] > data.max() / 2 else "black"
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", color=color)
    fig.tight_layout()
    return fig


model = load_model()
metrics = load_metrics()
classes = metrics["classes"]

st.title("🎙️ Speech Emotion Detection")
st.caption("Upload a short speech clip (WAV) and the model predicts the emotion: angry, happy, sad or neutral.")

tab_predict, tab_results, tab_about = st.tabs(["Predict", "Results", "About"])

with tab_predict:
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded is None:
        st.info("Upload a WAV file to see the predicted emotion, probabilities and waveform.")
    else:
        file_bytes = uploaded.getvalue()
        st.audio(file_bytes, format="audio/wav")
        try:
            proba = predict_proba(file_bytes, model, metrics["feature_indices"])
            fig, duration = plot_waveform(file_bytes, metrics["sample_rate"])
        except Exception as e:
            st.error(f"Could not process this file. Please upload a valid WAV recording. ({e})")
        else:
            top = proba.idxmax()
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Predicted emotion")
                st.markdown(f"# {EMOJI.get(top, '')} {top.capitalize()}")
                st.metric("Confidence", f"{proba.max():.0%}")
                if proba.max() < 0.5:
                    st.warning("Low confidence: the model is unsure about this clip.")
                if duration > 10:
                    st.warning("This clip is long. The model was trained on clips of about 3-4 seconds.")
            with col2:
                st.subheader("Probability per emotion")
                chart_df = proba.reindex(classes).rename("probability").to_frame()
                st.bar_chart(chart_df, y="probability", horizontal=True)

            st.subheader("Waveform")
            st.pyplot(fig)

with tab_results:
    st.subheader("Test results on unseen actors")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    c3.metric("Test clips", metrics["n_test"])
    st.caption(
        f"Model: {metrics['model']} | Features: {metrics['feature_set']} | "
        f"Training: {metrics['training_regime']} | "
        f"Train actors {min(metrics['train_actors'])}-{max(metrics['train_actors'])}, "
        f"test actors {min(metrics['test_actors'])}-{max(metrics['test_actors'])}"
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Class counts (4 emotions, 672 clips)")
        counts = pd.Series(metrics["class_counts_all"]).reindex(classes).rename("clips").to_frame()
        st.bar_chart(counts, y="clips")
    with right:
        st.subheader("Per-class scores")
        report = metrics["per_class_report"]
        table = pd.DataFrame(
            {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in classes}
        ).T
        st.dataframe(table.round(3))

    st.subheader("Confusion matrix")
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
noisy audio or other accents and languages will be lower.
"""
    )
