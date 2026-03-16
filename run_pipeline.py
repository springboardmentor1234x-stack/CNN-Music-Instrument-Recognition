
import os
from src.preprocess import load_audio, normalize_audio, trim_silence
from src.spectrogram import generate_mel_spectrogram
from src.visualize import plot_spectrogram
from src.predict import predict_from_spectrogram

audio_path = "data/raw_audio/sample.wav"

audio,sr = load_audio(audio_path)

audio = normalize_audio(audio)

audio = trim_silence(audio)

spec = generate_mel_spectrogram(audio,sr)

plot_spectrogram(spec)

predict_from_spectrogram(spec)
