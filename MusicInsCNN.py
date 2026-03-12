# ===============================
# 1. Import Libraries
# ===============================

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models


# ===============================
# 2. Dataset Location
# ===============================

DATA_DIR = "D:\MusicInstrumentCNN\IRMAS-TrainingData"


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


# ===============================
# 5. Audio Augmentation
# ===============================

def audio_augmentation(signal, sr):

    if np.random.rand() > 0.5:
        signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=2)

    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.002, len(signal))
        signal = signal + noise

    return signal


# ===============================
# 6. Feature Extraction
# ===============================

def extract_features(path, augment=False):

    signal, sr = librosa.load(path, sr=16000)

    if augment:
        signal = audio_augmentation(signal, sr)

    signal, _ = librosa.effects.trim(signal)

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_db = (mel_db - np.mean(mel_db)) / np.std(mel_db)

    target_frames = 128

    if mel_db.shape[1] < target_frames:

        padding = target_frames - mel_db.shape[1]

        mel_db = np.pad(mel_db, ((0,0),(0,padding)), mode="constant")

    else:

        mel_db = mel_db[:, :target_frames]

    return mel_db


# ===============================
# 7. Train / Validation / Test Split
# ===============================

train_files, temp_files, train_labels, temp_labels = train_test_split(
    audio_files,
    encoded_labels,
    test_size=0.2,
    stratify=encoded_labels,
    random_state=42
)

val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=42
)

print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Test:", len(test_files))


# ===============================
# 8. Process Dataset
# ===============================

def build_dataset(file_list, label_list, augment=False):

    features = []
    targets = []

    for f, l in zip(file_list, label_list):

        spec = extract_features(f, augment)

        features.append(spec)
        targets.append(l)

    features = np.array(features)[..., np.newaxis]
    targets = np.array(targets)

    return features, targets


X_train, y_train = build_dataset(train_files, train_labels, augment=True)
X_val, y_val = build_dataset(val_files, val_labels)
X_test, y_test = build_dataset(test_files, test_labels)

print("Train shape:", X_train.shape)


# ===============================
# 9. CNN Architecture
# ===============================

model = models.Sequential([

    layers.Conv2D(32,(3,3),activation="relu",input_shape=(128,128,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(256,activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(len(class_names),activation="softmax")

])

model.summary()


# ===============================
# 10. Compile Model
# ===============================

model.compile(

    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]

)


# ===============================
# 11. Train Model
# ===============================

history = model.fit(

    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val,y_val)

)


# ===============================
# 12. Model Evaluation
# ===============================

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)


# ===============================
# 13. Predictions
# ===============================

pred_probs = model.predict(X_test)

pred_classes = np.argmax(pred_probs, axis=1)

print("Overall Accuracy:", accuracy_score(y_test, pred_classes))


# ===============================
# 14. Classification Report
# ===============================

print(classification_report(
    y_test,
    pred_classes,
    target_names=class_names
))


# ===============================
# 15. Confusion Matrix
# ===============================

cm = confusion_matrix(y_test, pred_classes)

import seaborn as sns

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# ===============================
# Training Graphs
# ===============================

plt.figure(figsize=(12,4))

# Accuracy Graph
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])

# Loss Graph
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Validation"])

plt.show()
# ===============================
# Training Graphs
# ===============================

plt.figure(figsize=(12,4))

# Accuracy Graph
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])

# Loss Graph
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Validation"])

plt.show()