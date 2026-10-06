import numpy as np
import librosa

SR = 22050
N_MFCC = 40

_PROSODY = [f"{n}_{s}" for n in ("zcr", "rms", "centroid", "rolloff", "bandwidth") for s in ("mean", "std")]
FEATURE_NAMES = ([f"mfcc_mean_{i}" for i in range(N_MFCC)] + [f"mfcc_std_{i}" for i in range(N_MFCC)]
                 + [f"delta_mean_{i}" for i in range(N_MFCC)] + [f"chroma_{i}" for i in range(12)]
                 + [f"mel_{i}" for i in range(64)] + [f"contrast_{i}" for i in range(7)] + _PROSODY)

def load_audio(path_or_file, sr=SR, trim_db=30):
    """Load audio, trim leading/trailing silence, guarantee a minimum length."""
    y, _ = librosa.load(path_or_file, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    return librosa.util.fix_length(y, size=max(len(y), sr // 2))

def augment(y, kind, sr=SR, seed=0):
    """Simple augmentations (used on training actors only)."""
    if kind == "noise":
        rng = np.random.default_rng(seed)
        return y + rng.normal(0, 0.01 * (np.abs(y).max() + 1e-9), len(y))
    if kind == "stretch":
        return librosa.effects.time_stretch(y, rate=0.9)
    if kind == "pitch":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
    raise ValueError(kind)

def signal_features(y, sr=SR):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64), ref=1.0)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    prosody = []
    for m in (librosa.feature.zero_crossing_rate(y), librosa.feature.rms(y=y),
              librosa.feature.spectral_centroid(y=y, sr=sr),
              librosa.feature.spectral_rolloff(y=y, sr=sr),
              librosa.feature.spectral_bandwidth(y=y, sr=sr)):
        prosody += [m.mean(), m.std()]
    vec = np.concatenate([mfcc.mean(1), mfcc.std(1), delta.mean(1), chroma.mean(1),
                          mel.mean(1), contrast.mean(1), np.array(prosody)])
    assert len(vec) == len(FEATURE_NAMES)
    return vec

def extract_features(path_or_file, sr=SR):
    """Full feature vector for one audio file (used by the dashboard)."""
    return signal_features(load_audio(path_or_file, sr), sr)
