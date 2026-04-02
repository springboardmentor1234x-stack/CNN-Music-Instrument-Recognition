import os
import numpy as np
import librosa
import soundfile as sf
import json

def generate_sine_wave(freq, duration, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return wave

def generate_square_wave(freq, duration, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.5 * np.sign(np.sin(2 * np.pi * freq * t))
    return wave

def generate_sawtooth_wave(freq, duration, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.5 * (2 * (t * freq - np.floor(0.5 + t * freq)))
    return wave

def generate_synthetic_dataset(output_dir, num_samples_per_class=50, duration=3.0, sr=22050):
    """
    Generates a tiny synthetic dataset of simple waveforms to act as a stand-in for musical instruments.
    Helps rapidly test the pipeline locally.
    
    Classes:
    0 - Sine Wave (e.g. Flute-like)
    1 - Square Wave (e.g. Clarinet-like)
    2 - Sawtooth Wave (e.g. Brass-like)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    classes = ['sine', 'square', 'sawtooth']
    generators = [generate_sine_wave, generate_square_wave, generate_sawtooth_wave]
    
    metadata = []
    
    print(f"Generating synthetic dataset in '{output_dir}'...")
    
    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(num_samples_per_class):
            freq = np.random.uniform(200, 800) # Random fundamental frequency
            wave = generators[class_id](freq, duration, sr)
            
            # Add a little white noise
            noise = np.random.normal(0, 0.05, wave.shape)
            wave = wave + noise
            
            # Normalize
            if np.max(np.abs(wave)) > 0:
                wave = wave / np.max(np.abs(wave))
                
            filename = f"{class_name}_{i:03d}.wav"
            filepath = os.path.join(class_dir, filename)
            
            sf.write(filepath, wave, sr)
            
            # Since the project asks for multi-class OR multi-label, we'll start with multi-class
            # but support multi-label by making the label an array of presence.
            # Example multi-label [sine_present, square_present, sawtooth_present]
            labels = [0, 0, 0]
            labels[class_id] = 1
            
            metadata.append({
                "filename": filename,
                "filepath": filepath,
                "class_name": class_name,
                "labels": labels
            })
            
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Generated {len(metadata)} samples across {len(classes)} classes.")

if __name__ == "__main__":
    import sys
    
    output_dir = "data/synthetic_dataset"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        
    generate_synthetic_dataset(output_dir)
