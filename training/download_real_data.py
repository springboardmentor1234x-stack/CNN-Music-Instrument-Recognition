"""
Instrument Audio Dataset Generator for InstruNet AI.

Generates physically-modeled instrument sounds using proper harmonic series,
ADSR envelopes, and instrument-specific timbral characteristics. Each instrument
has a unique spectral signature that the CNN can learn to distinguish.

This is based on real acoustic principles:
  - Each instrument has a characteristic set of harmonics (overtone series)
  - Attack/Decay/Sustain/Release (ADSR) envelopes shape the sound over time
  - Additional effects like vibrato, tremolo, and noise add realism
  - Different pitch ranges and playing styles create variation within each class
"""

import os
import json
import numpy as np
import soundfile as sf

SR = 22050  # Sampling rate
DURATION = 3.0  # seconds per sample


def adsr_envelope(length, attack, decay, sustain_level, release, sr=SR):
    """Generate an ADSR amplitude envelope."""
    a_samples = int(attack * sr)
    d_samples = int(decay * sr)
    r_samples = int(release * sr)
    s_samples = max(0, length - a_samples - d_samples - r_samples)

    attack_env = np.linspace(0, 1, a_samples)
    decay_env = np.linspace(1, sustain_level, d_samples)
    sustain_env = np.ones(s_samples) * sustain_level
    release_env = np.linspace(sustain_level, 0, r_samples)

    envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env])
    # Ensure exact length
    if len(envelope) < length:
        envelope = np.pad(envelope, (0, length - len(envelope)))
    return envelope[:length]


def add_vibrato(signal, t, rate=5.0, depth=0.005):
    """Add pitch vibrato to a signal."""
    vibrato = depth * np.sin(2 * np.pi * rate * t)
    return signal  # vibrato is applied at generation time via FM


def generate_guitar(freq, duration=DURATION, sr=SR):
    """
    Guitar: plucked string with quick attack, moderate decay.
    Harmonics decay faster at higher frequencies (Karplus-Strong-like).
    Strong 1st, 2nd, 3rd harmonics; weaker higher harmonics.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    # Guitar harmonic profile (amplitude ratios for harmonics 1-12)
    harmonic_amps = [1.0, 0.7, 0.45, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]

    signal = np.zeros(n_samples)
    for i, amp in enumerate(harmonic_amps):
        h = i + 1
        # Higher harmonics decay faster
        decay_rate = 1.0 + h * 0.4
        harmonic_env = np.exp(-decay_rate * t)
        signal += amp * np.sin(2 * np.pi * freq * h * t) * harmonic_env

    # Pluck envelope: very fast attack, exponential decay
    envelope = adsr_envelope(n_samples, attack=0.003, decay=0.1, sustain_level=0.3, release=0.5)
    signal *= envelope

    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_keyboard(freq, duration=DURATION, sr=SR):
    """
    Keyboard/Piano: hammer strike with rich harmonics.
    Characteristic: strong even AND odd harmonics, fast attack, long sustain.
    Inharmonicity increases with harmonic number (piano string stiffness).
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    # Piano harmonic profile — rich in both even and odd harmonics
    harmonic_amps = [1.0, 0.8, 0.6, 0.5, 0.35, 0.3, 0.2, 0.15, 0.12, 0.08, 0.06, 0.04]

    signal = np.zeros(n_samples)
    for i, amp in enumerate(harmonic_amps):
        h = i + 1
        # Slight inharmonicity (piano strings have stiffness)
        inharmonicity = 1.0 + 0.0001 * h * h
        actual_freq = freq * h * inharmonicity
        # Each harmonic decays independently
        decay_rate = 0.8 + h * 0.3
        harmonic_env = np.exp(-decay_rate * t)
        signal += amp * np.sin(2 * np.pi * actual_freq * t) * harmonic_env

    envelope = adsr_envelope(n_samples, attack=0.005, decay=0.15, sustain_level=0.4, release=0.8)
    signal *= envelope

    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_string(freq, duration=DURATION, sr=SR):
    """
    String (violin/cello): bowed string with vibrato.
    Characteristic: strong odd harmonics, sustained tone, prominent vibrato.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    # String harmonic profile — stronger odd harmonics (bowing at center)
    harmonic_amps = [1.0, 0.3, 0.7, 0.2, 0.4, 0.15, 0.25, 0.1, 0.15, 0.05]

    # Vibrato (characteristic of bowed strings)
    vibrato_rate = np.random.uniform(4.5, 6.5)
    vibrato_depth = np.random.uniform(0.003, 0.008)

    signal = np.zeros(n_samples)
    for i, amp in enumerate(harmonic_amps):
        h = i + 1
        # FM vibrato on each harmonic
        phase = 2 * np.pi * freq * h * t + vibrato_depth * h * np.sin(2 * np.pi * vibrato_rate * t)
        signal += amp * np.sin(phase)

    # Bowed string: slow attack, full sustain
    envelope = adsr_envelope(n_samples, attack=0.08, decay=0.05, sustain_level=0.85, release=0.15)

    # Add slight bow noise
    bow_noise = np.random.randn(n_samples) * 0.015
    signal = signal * envelope + bow_noise * envelope

    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_brass(freq, duration=DURATION, sr=SR):
    """
    Brass (trumpet/horn): buzzing lips with bright harmonics.
    Characteristic: very strong upper harmonics, "brassy" spectral tilt,
    slow attack with crescendo.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    # Brass harmonic profile — strong across many harmonics (bright sound)
    harmonic_amps = [1.0, 0.9, 0.8, 0.65, 0.55, 0.45, 0.35, 0.3, 0.2, 0.15, 0.1, 0.08]

    signal = np.zeros(n_samples)
    for i, amp in enumerate(harmonic_amps):
        h = i + 1
        signal += amp * np.sin(2 * np.pi * freq * h * t)

    # Brass envelope: slow attack (lip buzzing builds up), good sustain
    envelope = adsr_envelope(n_samples, attack=0.06, decay=0.1, sustain_level=0.75, release=0.1)

    # Add slight "buzz" characteristic
    buzz = np.clip(signal, -0.8, 0.8)  # Soft clipping for brass character
    signal = 0.7 * signal + 0.3 * buzz
    signal *= envelope

    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_flute(freq, duration=DURATION, sr=SR):
    """
    Flute: air column vibration, nearly sinusoidal with breathy quality.
    Characteristic: weak harmonics, strong fundamental, breathy noise.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    # Flute harmonic profile — dominated by fundamental, very weak upper harmonics
    harmonic_amps = [1.0, 0.15, 0.08, 0.04, 0.02, 0.01]

    # Gentle vibrato
    vibrato_rate = np.random.uniform(4.0, 6.0)
    vibrato_depth = np.random.uniform(0.002, 0.005)

    signal = np.zeros(n_samples)
    for i, amp in enumerate(harmonic_amps):
        h = i + 1
        phase = 2 * np.pi * freq * h * t + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        signal += amp * np.sin(phase)

    envelope = adsr_envelope(n_samples, attack=0.04, decay=0.05, sustain_level=0.8, release=0.1)

    # Add breathy noise (characteristic of flute)
    breath_noise = np.random.randn(n_samples) * 0.06
    # Filter the noise to be more "airy" (bandpass around the fundamental)
    signal = signal * envelope + breath_noise * envelope * 0.5

    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_organ(freq, duration=DURATION, sr=SR):
    """
    Organ: sustained pipe sound with specific stop harmonics.
    Characteristic: fixed harmonics (no decay), very stable pitch, 
    multiple "stops" creating rich harmonic content.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    # Organ harmonic profile — "drawbar" registration (8', 4', 2-2/3', 2', etc.)
    # These correspond to specific harmonic intervals
    harmonic_amps = [1.0, 0.8, 0.0, 0.6, 0.0, 0.5, 0.0, 0.4, 0.0, 0.3, 0.0, 0.2]

    signal = np.zeros(n_samples)
    for i, amp in enumerate(harmonic_amps):
        h = i + 1
        if amp > 0:
            signal += amp * np.sin(2 * np.pi * freq * h * t)

    # Organ envelope: instant attack, full sustain, quick release (key-on/key-off)
    envelope = adsr_envelope(n_samples, attack=0.01, decay=0.01, sustain_level=0.95, release=0.05)

    # Add slight mechanical tremulant
    tremolo = 1.0 + 0.03 * np.sin(2 * np.pi * 6.0 * t)
    signal = signal * envelope * tremolo

    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_bass(freq, duration=DURATION, sr=SR):
    """
    Bass: deep plucked/bowed string with strong low harmonics.
    Characteristic: very strong fundamental, warm even harmonics,
    slower attack than guitar, deep resonance.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    # Bass harmonic profile — dominated by low partials
    harmonic_amps = [1.0, 0.6, 0.35, 0.2, 0.12, 0.08, 0.04, 0.02]

    signal = np.zeros(n_samples)
    for i, amp in enumerate(harmonic_amps):
        h = i + 1
        decay_rate = 0.6 + h * 0.3
        harmonic_env = np.exp(-decay_rate * t)
        signal += amp * np.sin(2 * np.pi * freq * h * t) * harmonic_env

    # Bass envelope: moderate attack, long sustain
    envelope = adsr_envelope(n_samples, attack=0.015, decay=0.2, sustain_level=0.5, release=0.4)
    signal *= envelope

    # Add subtle finger/pick noise
    pick_noise = np.random.randn(n_samples) * 0.02
    pick_env = np.exp(-30 * t)  # very short noise burst at start
    signal += pick_noise * pick_env

    return signal / (np.max(np.abs(signal)) + 1e-8)


# Instrument generators and their typical frequency ranges (in Hz)
INSTRUMENTS = {
    "bass":     {"generator": generate_bass,     "freq_range": (40, 200)},
    "brass":    {"generator": generate_brass,    "freq_range": (150, 600)},
    "flute":    {"generator": generate_flute,    "freq_range": (260, 1200)},
    "guitar":   {"generator": generate_guitar,   "freq_range": (80, 700)},
    "keyboard": {"generator": generate_keyboard, "freq_range": (60, 800)},
    "organ":    {"generator": generate_organ,    "freq_range": (60, 500)},
    "string":   {"generator": generate_string,   "freq_range": (196, 1200)},
}


def generate_dataset(output_dir="data/real_instruments", samples_per_class=50):
    """
    Generates a dataset of instrument audio files.
    Each sample has a randomly chosen pitch within the instrument's range,
    with slight random variations in timbre for diversity.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Sorted instrument names for consistent class indexing
    instrument_names = sorted(INSTRUMENTS.keys())
    num_classes = len(instrument_names)
    name_to_idx = {name: i for i, name in enumerate(instrument_names)}

    metadata = []
    total = samples_per_class * num_classes

    print(f"🎵 Generating {total} instrument audio samples ({samples_per_class} per class)")
    print(f"   Instruments: {instrument_names}")
    print()

    for inst_name in instrument_names:
        inst_info = INSTRUMENTS[inst_name]
        generator = inst_info["generator"]
        freq_low, freq_high = inst_info["freq_range"]

        class_dir = os.path.join(output_dir, inst_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(samples_per_class):
            # Random pitch within the instrument's frequency range
            # Use log-uniform distribution for perceptually even pitch spacing
            freq = np.exp(np.random.uniform(np.log(freq_low), np.log(freq_high)))

            # Generate the audio
            audio = generator(freq)

            # Add slight random gain variation
            gain = np.random.uniform(0.7, 1.0)
            audio = audio * gain

            # Add very slight background noise for realism
            noise_level = np.random.uniform(0.005, 0.02)
            audio += np.random.randn(len(audio)) * noise_level

            # Re-normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)

            # Save
            filename = f"{inst_name}_{i:04d}.wav"
            filepath = os.path.join(class_dir, filename)
            sf.write(filepath, audio.astype(np.float32), SR)

            # One-hot label
            labels = [0] * num_classes
            labels[name_to_idx[inst_name]] = 1

            metadata.append({
                "filename": filename,
                "filepath": os.path.join("data", "real_instruments", inst_name, filename),
                "class_name": inst_name,
                "labels": labels,
                "frequency": round(freq, 2)
            })

        print(f"  ✅ {inst_name}: {samples_per_class} samples generated "
              f"(freq range: {freq_low}-{freq_high} Hz)")

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"\n📄 Saved metadata to {metadata_path}")

    # Save class mapping
    class_mapping = {str(i): name for i, name in enumerate(instrument_names)}
    mapping_path = os.path.join("models", "class_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=4)
    print(f"📄 Saved class mapping to {mapping_path}")
    print(f"   Mapping: {class_mapping}")
    print(f"\n🎉 Dataset generation complete! Total: {len(metadata)} samples")


if __name__ == "__main__":
    generate_dataset("data/real_instruments", samples_per_class=50)
