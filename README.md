# Speech Emotion Detection (RAVDESS)

Predicts the emotion of a short speech clip: **angry, happy, sad or neutral**.
Built with librosa, scikit-learn and Streamlit.

**Live dashboard:** <PASTE_YOUR_LINK_HERE>

## Results (test actors 19-24, never seen in training)

| Model | Features | Accuracy | Macro F1 |
|---|---|---|---|
| <MODEL> | <FEATURES> | <ACC> | <F1> |

Confusion matrix: `confusion_matrix.png`. Full metrics: `metrics.json`.

## Data
RAVDESS speech audio (Kaggle: `uwrfkaggler/ravdess-emotional-speech-audio`). Only 4 emotions are used:
672 clips (192 each for angry, happy, sad and 96 for neutral) from 24 actors.

## Method
1. Filter speech clips and 4 emotions, remove duplicate files.
2. Explore class counts, waveforms and mel spectrograms.
3. Extract 213 audio features per clip with librosa (MFCC, delta-MFCC, chroma, log-mel, spectral contrast, loudness and brightness statistics). Silence is trimmed first.
4. Split **by actor**: actors 1-18 for training, actors 19-24 for testing, so the model is tested on voices it has never heard (no data leakage).
5. Train SVM and Random Forest with `class_weight="balanced"` because the neutral class is half the size of the others.
6. Tune with actor-grouped 5-fold cross-validation on the training actors. The final model is chosen by cross-validation macro F1, not by test score.
7. Augmentation (noise, time-stretch, pitch-shift) is applied to training actors only.

## Files
| File | Purpose |
|---|---|
| `speech_emotion_detection.ipynb` | Full pipeline: data, exploration, features, training, evaluation |
| `features.py` | Feature extraction, shared by the notebook and the dashboard |
| `app.py` | Streamlit dashboard |
| `model.joblib` | Trained model |
| `metrics.json` | Saved metrics used by the results page |
| `requirements.txt` | Python dependencies |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
RAVDESS is acted speech recorded in a studio by 24 North American actors. Performance on real phone calls,
noisy audio, other accents or other languages will be lower.

## Improvements with more time
Pretrained speech embeddings (wav2vec2 / HuBERT), leave-one-actor-out evaluation, more augmentation,
all 8 emotions, and testing on other emotional speech datasets.
