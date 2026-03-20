
import librosa
import numpy as np

def generate_mel_spectrogram(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        hop_length=512,
        n_fft=2048
    )

    log_mel = librosa.power_to_db(mel)
    return log_mel
