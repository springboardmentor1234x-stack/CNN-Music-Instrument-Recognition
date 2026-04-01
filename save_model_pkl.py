# ============================================
# Save Trained Model as PKL for Streamlit Deployment
# Bundles: model weights, architecture config,
#   label encoder, preprocessing params, class labels,
#   evaluation results, and instrument name mapping
# ============================================

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.preprocessing import LabelEncoder

# ===============================
# 1. Paths
# ===============================
MODEL_DIR = r"D:\AI ONBOARDING ENGINE TT\cnn_music_instrument_recognition\models"
PTH_PATH = os.path.join(MODEL_DIR, "instrument_classifier_best.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "label_classes.json")
EVAL_PATH = os.path.join(MODEL_DIR, "evaluation_results.json")
PKL_OUTPUT = os.path.join(MODEL_DIR, "instrument_classifier_full.pkl")

# ===============================
# 2. Load class labels
# ===============================
with open(LABELS_PATH, "r") as f:
    class_names = json.load(f)

print(f"Classes ({len(class_names)}): {class_names}")

# ===============================
# 3. Rebuild LabelEncoder
# ===============================
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(class_names)

# ===============================
# 4. Load evaluation results
# ===============================
eval_results = {}
if os.path.exists(EVAL_PATH):
    with open(EVAL_PATH, "r") as f:
        eval_results = json.load(f)
    print(f"Evaluation results loaded (accuracy: {eval_results.get('accuracy', 'N/A')})")

# ===============================
# 5. Rebuild model architecture
# ===============================
device = torch.device("cpu")  # Save on CPU for portability

model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(class_names))
)

# Load trained weights
model.load_state_dict(torch.load(PTH_PATH, map_location=device))
model.eval()
print("Model weights loaded successfully.")

# ===============================
# 6. Instrument name mapping
# ===============================
INSTRUMENT_NAMES = {
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

# ===============================
# 7. Preprocessing config
# ===============================
preprocessing_config = {
    "sample_rate": 22050,
    "duration_seconds": 3,
    "n_mels": 224,
    "hop_length": 256,
    "target_shape": (224, 224),
    "normalization": "z-score",       # (x - mean) / (std + 1e-8)
    "channels": 3,                     # mono spectrogram stacked 3x for EfficientNet
    "power_to_db_ref": "np.max",
}

# ===============================
# 8. Model architecture config
# ===============================
architecture_config = {
    "backbone": "EfficientNet-B0",
    "classifier": [
        {"type": "Dropout", "p": 0.4},
        {"type": "Linear", "in_features": num_features, "out_features": 256},
        {"type": "ReLU"},
        {"type": "Dropout", "p": 0.3},
        {"type": "Linear", "in_features": 256, "out_features": len(class_names)},
    ],
    "num_classes": len(class_names),
    "input_shape": (3, 224, 224),
}

# ===============================
# 9. Bundle everything into PKL
# ===============================
model_bundle = {
    # --- Model ---
    "model_state_dict": model.state_dict(),
    "architecture_config": architecture_config,

    # --- Labels & Encoding ---
    "class_names": class_names,
    "label_encoder": label_encoder,
    "instrument_names": INSTRUMENT_NAMES,

    # --- Preprocessing ---
    "preprocessing_config": preprocessing_config,

    # --- Evaluation Metrics ---
    "evaluation_results": eval_results,

    # --- Metadata ---
    "metadata": {
        "framework": "PyTorch",
        "model_name": "InstruNet-EfficientNet-B0",
        "version": "1.0",
        "description": "CNN-based Music Instrument Recognition using EfficientNet-B0 backbone",
        "source_pth": "instrument_classifier_best.pth",
    }
}

# ===============================
# 10. Save PKL
# ===============================
with open(PKL_OUTPUT, "wb") as f:
    pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

file_size_mb = os.path.getsize(PKL_OUTPUT) / (1024 * 1024)
print(f"\n{'=' * 55}")
print(f"  MODEL BUNDLE SAVED SUCCESSFULLY")
print(f"{'=' * 55}")
print(f"  Output: {PKL_OUTPUT}")
print(f"  Size:   {file_size_mb:.2f} MB")
print(f"  Keys:   {list(model_bundle.keys())}")
print(f"{'=' * 55}")

# ===============================
# 11. Verification - Load it back
# ===============================
print("\nVerifying PKL file...")
with open(PKL_OUTPUT, "rb") as f:
    loaded = pickle.load(f)

print(f"  ✅ Keys found:       {list(loaded.keys())}")
print(f"  ✅ Classes:          {loaded['class_names']}")
print(f"  ✅ Preprocessing:    {loaded['preprocessing_config']}")
print(f"  ✅ Architecture:     {loaded['architecture_config']['backbone']}")
print(f"  ✅ Num classes:      {loaded['architecture_config']['num_classes']}")
print(f"  ✅ Instrument map:   {len(loaded['instrument_names'])} instruments")
print(f"  ✅ Has eval results: {bool(loaded['evaluation_results'])}")

# Quick model rebuild test
test_model = models.efficientnet_b0(weights=None)
test_model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, loaded['architecture_config']['num_classes'])
)
test_model.load_state_dict(loaded['model_state_dict'])
test_model.eval()
print(f"  ✅ Model rebuilt from PKL successfully!")
print(f"\nDone. You can now use '{os.path.basename(PKL_OUTPUT)}' for Streamlit deployment.")