"""
CNN-Based Music Instrument Recognition System
Phase 1 - Advanced Audio Preprocessing & Augmentation
Creates high-quality Mel-Spectrogram images from IRMAS audio files.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IRMAS_TRAIN_PATH = r"data/IRMAS-TrainingData"
OUTPUT_PATH      = "data/spectrograms"

SAMPLE_RATE      = 22050
DURATION         = 3               # seconds per clip
N_MELS           = 224             # high-resolution mel
HOP_LENGTH       = 256
N_FFT            = 2048
IMG_SIZE         = (224, 224)

# Number of augmented versions per original file
AUGMENT_COPIES   = 2

INSTRUMENT_LABELS = {
    "cel": "Cello",
    "cla": "Clarinet",
    "flu": "Flute",
    "gac": "Acoustic Guitar",
    "gel": "Electric Guitar",
    "org": "Organ",
    "pia": "Piano",
    "sax": "Saxophone",
    "tru": "Trumpet",
    "vio": "Violin",
    "voi": "Voice",
}

# ─────────────────────────────────────────────
# STEP 1: Load audio
# ─────────────────────────────────────────────
def load_audio(filepath, sr=SAMPLE_RATE, duration=DURATION):
    """Load audio, trim silence, fix duration, normalize, pre-emphasis."""
    audio, _ = librosa.load(filepath, sr=sr, mono=True)

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Fix duration
    target_len = sr * duration
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # Normalize waveform
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    return audio


# ─────────────────────────────────────────────
# STEP 2: Audio Augmentations
# ─────────────────────────────────────────────
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return np.clip(audio + noise_factor * noise, -1.0, 1.0)

def time_shift(audio, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)

def pitch_shift(audio, sr=SAMPLE_RATE, n_steps=None):
    if n_steps is None:
        n_steps = np.random.uniform(-1.5, 1.5)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate=None):
    if rate is None:
        rate = np.random.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(audio, rate=rate)

    # Fix length after stretch
    target_len = SAMPLE_RATE * DURATION
    if len(stretched) < target_len:
        stretched = np.pad(stretched, (0, target_len - len(stretched)))
    else:
        stretched = stretched[:target_len]

    return stretched

def augment_audio(audio):
    """Randomly apply one or more augmentations."""
    aug_audio = audio.copy()

    if np.random.rand() > 0.5:
        aug_audio = add_noise(aug_audio)

    if np.random.rand() > 0.5:
        aug_audio = time_shift(aug_audio)

    if np.random.rand() > 0.5:
        aug_audio = pitch_shift(aug_audio)

    if np.random.rand() > 0.5:
        aug_audio = time_stretch(aug_audio)

    return aug_audio


# ─────────────────────────────────────────────
# STEP 3: Convert to Mel-Spectrogram
# ─────────────────────────────────────────────
def audio_to_melspectrogram(audio, sr=SAMPLE_RATE):
    """Convert audio to normalized Mel-Spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to 0–1
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    return mel_db


# ─────────────────────────────────────────────
# STEP 4: Save Spectrogram
# ─────────────────────────────────────────────
def save_spectrogram_image(mel_spec, save_path):
    """Save spectrogram as clean 224x224 PNG."""
    plt.figure(figsize=(4, 4), dpi=56)  # ~224x224
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ─────────────────────────────────────────────
# STEP 5: Process Dataset
# ─────────────────────────────────────────────
def preprocess_dataset(input_path=IRMAS_TRAIN_PATH, output_path=OUTPUT_PATH):
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {}

    for instrument_folder in sorted(input_path.iterdir()):
        if not instrument_folder.is_dir():
            continue

        code = instrument_folder.name.lower()[:3]
        label = INSTRUMENT_LABELS.get(code, code)

        save_dir = output_path / code
        save_dir.mkdir(parents=True, exist_ok=True)

        audio_files = list(instrument_folder.glob("*.wav")) + \
                      list(instrument_folder.glob("*.mp3")) + \
                      list(instrument_folder.glob("*.flac"))

        count = 0

        for audio_file in audio_files:
            try:
                # Original
                audio = load_audio(str(audio_file))
                mel_spec = audio_to_melspectrogram(audio)
                img_name = audio_file.stem + "_orig.png"
                save_spectrogram_image(mel_spec, str(save_dir / img_name))
                count += 1

                # Augmented copies
                for aug_idx in range(AUGMENT_COPIES):
                    aug_audio = augment_audio(audio)
                    aug_mel   = audio_to_melspectrogram(aug_audio)
                    aug_name  = f"{audio_file.stem}_aug{aug_idx+1}.png"
                    save_spectrogram_image(aug_mel, str(save_dir / aug_name))
                    count += 1

                if count % 100 == 0:
                    print(f"  [{label}] Generated {count} spectrograms...")

            except Exception as e:
                print(f"  ERROR: {audio_file.name} — {e}")

        stats[code] = {"label": label, "count": count}
        print(f"✅ {label}: {count} spectrograms saved → {save_dir}")

    # Save dataset stats
    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n📊 Stats saved to {stats_file}")
    return stats


# ─────────────────────────────────────────────
# STEP 6: Visual Check
# ─────────────────────────────────────────────
def visualize_sample(audio_path):
    audio = load_audio(audio_path)
    mel   = audio_to_melspectrogram(audio)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    librosa.display.waveshow(audio, sr=SAMPLE_RATE, ax=axes[0], color="#e63946")
    axes[0].set_title("Waveform", fontsize=13)

    img = librosa.display.specshow(
        mel,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap="magma"
    )
    fig.colorbar(img, ax=axes[1])
    axes[1].set_title("Advanced Mel-Spectrogram", fontsize=13)

    plt.tight_layout()
    plt.savefig("sample_visualization.png", dpi=150)
    plt.show()
    print("Sample visualization saved as sample_visualization.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CNN Music Instrument Recognition — Phase 1 Advanced")
    print("=" * 60)

    stats = preprocess_dataset()

    print("\n📁 All done! Advanced spectrogram dataset created.")
    print(f"   Total instruments: {len(stats)}")
    print(f"   Total files: {sum(v['count'] for v in stats.values())}")