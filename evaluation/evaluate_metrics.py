import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.audio_processor import load_and_preprocess_audio, generate_mel_spectrogram

def load_dataset_from_metadata(metadata_path, base_dir):
    """
    Loads audio files, converts them to spectrograms, and prepares the labels.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    X = []
    y = []
    skipped = 0
    
    print(f"Loading {len(metadata)} audio files for evaluation...")
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
            
        # Crop/pad to exactly 3 seconds
        target_length = sr * 3
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            
        spec = generate_mel_spectrogram(audio, sr)  # Shape: (128, frames, 1)
        
        X.append(spec)
        y.append(labels)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(metadata)} files...")
            
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} files (missing or unreadable)")
        
    return np.array(X), np.array(y)

def run_evaluation():
    # Paths
    dataset_dir = "data/real_instruments"
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    class_mapping_path = os.path.join("models", "class_mapping.json")
    model_path = os.path.join("models", "baseline_cnn.h5")
    
    # Check if files exist
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found at {metadata_path}")
        return
    if not os.path.exists(class_mapping_path):
        print(f"❌ Class mapping not found at {class_mapping_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    target_names = [class_mapping[str(i)] for i in range(num_classes)]

    # 1. Load data
    X, y = load_dataset_from_metadata(metadata_path, base_dir=".")
    print(f"\nTotal Dataset shape: X={X.shape}, y={y.shape}")
    
    # 2. Split (Same as train.py to isolate validation set)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    print(f"Evaluating on {len(X_test)} validation samples...")
    
    # 3. Load model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 4. Predict
    print("Running inference...")
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 5. Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    print("\n" + "═"*50)
    print(" " * 15 + "MODEL PERFORMANCE METRICS")
    print("═"*50)
    print(f"OVERALL ACCURACY: {accuracy*100:.2f}%")
    print("-" * 50)
    
    # Format report into a readable table
    report_df = pd.DataFrame(report).transpose()
    # Remove 'accuracy', 'macro avg', 'weighted avg' rows for the per-class table
    per_class_metrics = report_df.iloc[:-3, :3] 
    print(per_class_metrics)
    print("-" * 50)
    
    # Summary of Mean Metrics
    print(f"Macro Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Recall:    {report['macro avg']['recall']:.4f}")
    print(f"Macro F1-Score:  {report['macro avg']['f1-score']:.4f}")
    print("═"*50)

    # 6. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    print("\nCONFUSION MATRIX:")
    print(cm_df)
    
    # Optional: Save results to JSON
    results = {
        "accuracy": accuracy,
        "per_class": report,
        "confusion_matrix": cm.tolist()
    }
    with open("evaluation/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Results saved to evaluation/evaluation_results.json")

if __name__ == "__main__":
    run_evaluation()
