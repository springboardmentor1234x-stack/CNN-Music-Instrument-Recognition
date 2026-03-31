import librosa
import numpy as np

def load_audio(path, sr=22050):
    audio, sr = librosa.load(path, sr=sr, mono=True)
    return audio, sr

def normalize_audio(audio):
    return audio / (np.max(np.abs(audio)) + 1e-6)

def trim_silence(audio):
    audio, _ = librosa.effects.trim(audio)
    return audio