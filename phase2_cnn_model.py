# ===============================
# CNN-Based Music Instrument Recognition System
# Phase 2 - Advanced Training with EfficientNet-B0 (PyTorch)
# ===============================

import os
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
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

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# 3. Collect Audio Files
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
# 4. Encode Labels
# ===============================
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(instrument_labels)
class_names = encoder.classes_
print("Classes:", class_names)

# Save class labels
with open(os.path.join(MODEL_DIR, "label_classes.json"), "w") as f:
    json.dump(class_names.tolist(), f)

# ===============================
# 5. Train / Val / Test Split
# ===============================
train_files, temp_files, train_labels, temp_labels = train_test_split(
    audio_files, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# ===============================
# 6. Feature Extraction + Cache
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

    # Ensure fixed duration (3 sec)
    target_len = sr * 3
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)))
    else:
        signal = signal[:target_len]

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=224,   # higher resolution
        hop_length=256
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

    # Force shape to 224x224
    if mel_db.shape[1] < 224:
        mel_db = np.pad(mel_db, ((0, 0), (0, 224 - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :224]

    np.save(cache_path, mel_db)
    return mel_db

# ===============================
# 7. SpecAugment-style Augmentation
# ===============================
def audio_augmentation(spec):
    spec = spec.copy()

    # Time shift
    if np.random.rand() > 0.5:
        shift = np.random.randint(-30, 30)
        spec = np.roll(spec, shift, axis=1)

    # Frequency masking
    if np.random.rand() > 0.5:
        f = np.random.randint(5, 30)
        f0 = np.random.randint(0, max(1, spec.shape[0] - f))
        spec[f0:f0+f, :] = 0

    # Time masking
    if np.random.rand() > 0.5:
        t = np.random.randint(5, 30)
        t0 = np.random.randint(0, max(1, spec.shape[1] - t))
        spec[:, t0:t0+t] = 0

    return spec

# ===============================
# 8. Pre-build Cache
# ===============================
all_files = list(set(train_files + val_files + test_files))
total = len(all_files)
print(f"\nBuilding cache for {total} files...")

for i, f in enumerate(all_files):
    extract_features(f)
    if (i + 1) % 200 == 0 or (i + 1) == total:
        print(f"  Cached {i+1}/{total}")

print("Cache ready!\n")

# ===============================
# 9. PyTorch Dataset
# ===============================
class InstrumentDataset(Dataset):
    def __init__(self, file_list, label_list, augment=False):
        self.file_list = file_list
        self.label_list = label_list
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        spec = extract_features(self.file_list[idx]).copy()

        if self.augment:
            spec = audio_augmentation(spec)

        # Convert 1-channel spectrogram -> 3 channels for EfficientNet
        spec = np.stack([spec, spec, spec], axis=0)  # (3, 224, 224)

        spec = torch.tensor(spec, dtype=torch.float32)
        label = torch.tensor(self.label_list[idx], dtype=torch.long)
        return spec, label

train_dataset = InstrumentDataset(train_files, train_labels, augment=True)
val_dataset   = InstrumentDataset(val_files,   val_labels, augment=False)
test_dataset  = InstrumentDataset(test_files,  test_labels, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# ===============================
# 10. Advanced Model - EfficientNet-B0
# ===============================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze first layers (optional but helps stability)
for param in model.features[:4].parameters():
    param.requires_grad = False

num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(class_names))
)

model = model.to(device)

# ===============================
# 11. Loss / Optimizer / Scheduler
# ===============================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.3, patience=2, min_lr=1e-6
)

# ===============================
# 12. Training Loop
# ===============================
EPOCHS   = 30
PATIENCE = 6
best_val_loss    = float("inf")
patience_counter = 0
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for specs, labels in train_loader:
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(labels)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += len(labels)

    train_loss /= total
    train_acc   = correct / total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)

            val_loss    += loss.item() * len(labels)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total   += len(labels)

    val_loss /= val_total
    val_acc   = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "instrument_classifier_best.pth"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "instrument_classifier_best.pth"), map_location=device))
print("Best model loaded.")

# ===============================
# 13. Evaluation
# ===============================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for specs, labels in test_loader:
        specs = specs.to(device)
        outputs = model(specs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

test_acc = accuracy_score(all_labels, all_preds)
print("\nTest Accuracy:", test_acc)
print(classification_report(all_labels, all_preds, target_names=class_names))

# Save results
results = {
    "test_accuracy": float(test_acc),
    "classification_report": classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
}
with open(os.path.join(MODEL_DIR, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=4)

# ===============================
# 14. Confusion Matrix
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

# ===============================
# 15. Training Graphs
# ===============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_acc"], label="Train")
plt.plot(history["val_acc"], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_loss"], label="Train")
plt.plot(history["val_loss"], label="Validation")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_history.png"))
plt.show()

# ===============================
# 16. Random Prediction Demo
# ===============================
model.eval()
sample_idx = np.random.randint(0, len(test_dataset))
spec, true_label = test_dataset[sample_idx]

with torch.no_grad():
    output = model(spec.unsqueeze(0).to(device))
    predicted = output.argmax(1).item()

print("\nActual Instrument:   ", class_names[true_label.item()])
print("Predicted Instrument:", class_names[predicted])