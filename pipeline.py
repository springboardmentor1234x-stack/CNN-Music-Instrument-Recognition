import numpy as np
from src.preprocess import load_audio, normalize_audio, trim_silence
from src.spectrogram import generate_mel_spectrogram
from src.predict import aggregate, apply_threshold


# ✅ Fix spectrogram size to 128x128
def fix_spectrogram(spec, target_size=(128, 128)):
    h, w = spec.shape

    # Pad if width is smaller
    if w < target_size[1]:
        pad_width = target_size[1] - w
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')

    # Crop if width is larger
    if w > target_size[1]:
        spec = spec[:, :target_size[1]]

    return spec


def run_pipeline(path):
    audio, sr = load_audio(path)

    # Preprocessing
    audio = normalize_audio(audio)
    audio = trim_silence(audio)

    # Segment audio (2 sec)
    seg_len = sr * 2
    segments = [
        audio[i:i+seg_len]
        for i in range(0, len(audio), seg_len)
        if len(audio[i:i+seg_len]) == seg_len
    ]

    preds = []

    for seg in segments:
        # Generate spectrogram
        spec = generate_mel_spectrogram(seg, sr)

        # ✅ Fix size
        spec = fix_spectrogram(spec)

        # Reshape for model
        spec = spec.reshape(1, 128, 128, 1)

        # Simulated predictions
        preds.append(np.random.rand(5))

    # Aggregate predictions
    avg = aggregate(preds)

    # Apply threshold
    result = apply_threshold(avg)

    return result