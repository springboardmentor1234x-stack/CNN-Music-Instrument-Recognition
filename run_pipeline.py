import os
from src.preprocess import load_audio, normalize_audio, trim_silence
from src.spectrogram import generate_mel_spectrogram
from src.visualize import plot_spectrogram
from src.predict import predict_from_spectrogram


def segment_audio(audio, sr, segment_duration=2):

    segment_length = sr * segment_duration
    segments = []

    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        if len(segment) == segment_length:
            segments.append(segment)

    return segments


audio_path = "data/raw_audio/sample.wav"

audio,sr = load_audio(audio_path)
audio = normalize_audio(audio)
audio = trim_silence(audio)

segments = segment_audio(audio, sr)

all_preds = []

for seg in segments:
    spec = generate_mel_spectrogram(seg, sr)
    plot_spectrogram(spec)

    pred = predict_from_spectrogram(spec)
    all_preds.append(list(pred.values()))

# Averaging
if len(all_preds) > 0:
    avg = sum([sum(p) for p in all_preds]) / len(all_preds)
    print("Average prediction score:", avg)