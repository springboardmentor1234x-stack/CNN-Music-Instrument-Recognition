import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.audio_processor import load_and_preprocess_audio, generate_mel_spectrogram
from models.cnn_model import build_cnn_model

def load_dataset_from_metadata(metadata_path, base_dir):
    """
    Loads audio files, converts them to spectrograms, and prepares the labels.
    Reads class mapping from the metadata instead of hardcoding it.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    X = []
    y = []
    skipped = 0
    
    print(f"Processing {len(metadata)} audio files...")
    for idx, item in enumerate(metadata):
        filepath = os.path.join(base_dir, item['filepath'])
        labels = item['labels']
        
        if not os.path.exists(filepath):
            skipped += 1
            continue
        
        # Load and preprocess
        audio, sr = load_and_preprocess_audio(filepath)
        if audio is None:
            skipped += 1
            continue
            
        # Crop/pad to exactly 3 seconds (NSynth samples are ~4s at 16kHz)
        target_length = sr * 3
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            
        spec = generate_mel_spectrogram(audio, sr)  # Shape: (128, frames, 1)
        
        X.append(spec)
        y.append(labels)
        
        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/{len(metadata)} files...")
            
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} files (missing or unreadable)")
        
    return np.array(X), np.array(y)

def train_model():
    dataset_dir = "data/real_instruments"
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    class_mapping_path = os.path.join("models", "class_mapping.json")
    
    if not os.path.exists(metadata_path):
        print("❌ Dataset not found! Please run download_real_data.py first.")
        return
        
    if not os.path.exists(class_mapping_path):
        print("❌ Class mapping not found! Please run download_real_data.py first.")
        return
    
    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    print(f"Classes ({num_classes}): {list(class_mapping.values())}")
    
    # 1. Load data
    X, y = load_dataset_from_metadata(metadata_path, base_dir=".")
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    
    if len(X) < 10:
        print("❌ Not enough data to train. Need at least 10 samples.")
        return
    
    input_shape = X.shape[1:]
    
    # 2. Split into train and validation sets (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Print class distribution
    train_classes = np.argmax(y_train, axis=1)
    val_classes = np.argmax(y_val, axis=1)
    print("\nClass distribution (train):")
    for ci in range(num_classes):
        name = class_mapping[str(ci)]
        count = np.sum(train_classes == ci)
        print(f"  {name}: {count}")
    
    # 3. Build model
    model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
    
    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
        )
    ]
    
    # 5. Train
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluate
    print("\n📊 Final Evaluation:")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc*100:.1f}%")
    
    # Per-class accuracy
    predictions = model.predict(X_val, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_val, axis=1)
    
    print("\n  Per-class accuracy:")
    for ci in range(num_classes):
        name = class_mapping[str(ci)]
        mask = true_classes == ci
        if np.sum(mask) > 0:
            acc = np.mean(pred_classes[mask] == ci)
            print(f"    {name}: {acc*100:.1f}% ({np.sum(mask)} samples)")
    
    # 7. Save Model
    model_save_path = os.path.join("models", "baseline_cnn.h5")
    model.save(model_save_path)
    print(f"\n✅ Model saved to {model_save_path}")
    print(f"   Class mapping: {class_mapping_path}")

if __name__ == "__main__":
    train_model()
