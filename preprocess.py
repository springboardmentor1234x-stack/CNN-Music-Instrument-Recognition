import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "..", "dataset", "train_audio")
SPEC_DIR = os.path.join(BASE_DIR, "..", "spectrograms", "train")

os.makedirs(SPEC_DIR, exist_ok=True)

print("Processing IRMAS training data...")

for instrument in os.listdir(AUDIO_DIR):
    instrument_path = os.path.join(AUDIO_DIR, instrument)

    if not os.path.isdir(instrument_path):
        continue

    print(f"Instrument: {instrument}")

    for file in os.listdir(instrument_path):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(instrument_path, file)

            # Load audio
            y, sr = librosa.load(file_path, sr=22050, mono=True)
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            # Mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            )

            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Save image
            plt.figure(figsize=(4, 4))
            plt.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
            plt.axis("off")

            img_name = f"{instrument}_{file.replace('.wav', '.png')}"
            save_path = os.path.join(SPEC_DIR, img_name)
            plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close()

print("✅ IRMAS spectrogram generation completed.")
