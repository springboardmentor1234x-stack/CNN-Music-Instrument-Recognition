# ============================================
# CNN-Based Music Instrument Recognition System
# Phase 3 - Evaluation & Visualization
# Works with final PyTorch EfficientNet model
# ============================================

import os
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 2. Paths
# ===============================
DATA_DIR  = r"D:\AI ONBOARDING ENGINE TT\cnn_music_instrument_recognition\data\IRMAS-TrainingData"
CACHE_DIR = r"D:\AI ONBOARDING ENGINE TT\cnn_music_instrument_recognition\cache"
MODEL_DIR = r"D:\AI ONBOARDING ENGINE TT\cnn_music_instrument_recognition\models"

MODEL_PATH = os.path.join(MODEL_DIR, "instrument_classifier_best.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "label_classes.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# 3. Load Class Labels
# ===============================
with open(LABELS_PATH, "r") as f:
    class_names = np.array(json.load(f))

print("Classes:", class_names)

# ===============================
# 4. Collect Audio Files
# ===============================
audio_files = []
instrument_labels = []

for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        for audio_file in os.listdir(folder_path):
            if audio_file.endswith(".wav"):
                audio_files.append(os.path.join(folder_path, audio_file))
                instrument_labels.append(folder)

print("Dataset size:", len(audio_files))

# ===============================
# 5. Encode Labels
# ===============================
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(instrument_labels)

# ===============================
# 6. Same split as Phase 2
# ===============================
train_files, temp_files, train_labels, temp_labels = train_test_split(
    audio_files, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# ===============================
# 7. Feature Extraction (same as Phase 2)
# ===============================
def get_cache_path(audio_path):
    rel = os.path.relpath(audio_path, DATA_DIR)
    cache_path = os.path.join(CACHE_DIR, rel.replace(".wav", ".npy"))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    return cache_path

def extract_features(path):
    cache_path = get_cache_path(path)

    if os.path.exists(cache_path):
        return np.load(cache_path)

    signal, sr = librosa.load(path, sr=22050)
    signal, _ = librosa.effects.trim(signal)

    target_len = sr * 3
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)))
    else:
        signal = signal[:target_len]

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=224,
        hop_length=256
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

    if mel_db.shape[1] < 224:
        mel_db = np.pad(mel_db, ((0, 0), (0, 224 - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :224]

    np.save(cache_path, mel_db)
    return mel_db

# ===============================
# 8. Dataset Class
# ===============================
class InstrumentDataset(Dataset):
    def __init__(self, file_list, label_list):
        self.file_list = file_list
        self.label_list = label_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        spec = extract_features(self.file_list[idx]).copy()
        spec = np.stack([spec, spec, spec], axis=0)  # 3-channel

        spec = torch.tensor(spec, dtype=torch.float32)
        label = torch.tensor(self.label_list[idx], dtype=torch.long)
        return spec, label

test_dataset = InstrumentDataset(test_files, test_labels)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# ===============================
# 9. Load Model
# ===============================
model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(class_names))
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully.")

# ===============================
# 10. Evaluation
# ===============================
all_preds, all_labels = [], []

with torch.no_grad():
    for specs, labels in test_loader:
        specs = specs.to(device)
        outputs = model(specs)
        preds = outputs.argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ===============================
# 11. Metrics
# ===============================
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted")
recall    = recall_score(all_labels, all_preds, average="weighted")
f1        = f1_score(all_labels, all_preds, average="weighted")

print("\n" + "=" * 55)
print("           FINAL EVALUATION METRICS")
print("=" * 55)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
print("\nPer-Class Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Save JSON results
results = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "classification_report": report
}

with open(os.path.join(MODEL_DIR, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved → {os.path.join(MODEL_DIR, 'evaluation_results.json')}")

# ===============================
# 12. Confusion Matrix
# ===============================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.show()

print(f"Confusion matrix saved → {os.path.join(MODEL_DIR, 'confusion_matrix.png')}")

# ===============================
# 13. Per-Class Accuracy Bar Chart
# ===============================
per_class_acc = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(12, 6))
sns.barplot(x=class_names, y=per_class_acc)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.xlabel("Instrument")
plt.title("Per-Class Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "per_class_accuracy.png"))
plt.show()

print(f"Per-class accuracy chart saved → {os.path.join(MODEL_DIR, 'per_class_accuracy.png')}")

# ===============================
# 14. Random Prediction Demo
# ===============================
sample_idx = np.random.randint(0, len(test_dataset))
spec, true_label = test_dataset[sample_idx]

with torch.no_grad():
    output = model(spec.unsqueeze(0).to(device))
    predicted = output.argmax(1).item()

print("\nRandom Prediction Demo")
print("Actual Instrument:   ", class_names[true_label.item()])
print("Predicted Instrument:", class_names[predicted])

print("\nPhase 3 Evaluation completed successfully.")