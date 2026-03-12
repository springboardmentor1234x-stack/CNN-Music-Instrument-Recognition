# =========================================================
# Musical Instrument Classification using CNN
# =========================================================

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# Dataset Path
# =========================================================

DATASET_PATH = "IRMAS-TrainingData"


# =========================================================
# Load Dataset
# =========================================================

audio_paths = []
labels = []

for instrument in os.listdir(DATASET_PATH):

    instrument_path = os.path.join(DATASET_PATH, instrument)

    if os.path.isdir(instrument_path):

        for file in os.listdir(instrument_path):

            if file.endswith(".wav"):

                audio_paths.append(os.path.join(instrument_path, file))
                labels.append(instrument)

print("Dataset size:", len(audio_paths))


# =========================================================
# Encode Labels
# =========================================================

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

class_names = encoder.classes_

print("Classes:", class_names)


# =========================================================
# Train / Validation / Test Split
# =========================================================

train_files, temp_files, train_labels, temp_labels = train_test_split(
    audio_paths,
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


# =========================================================
# Data Augmentation
# =========================================================

def augment_audio(signal, sr):

    if np.random.rand() > 0.5:
        signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=2)

    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.002, len(signal))
        signal = signal + noise

    if np.random.rand() > 0.5:
        signal = librosa.effects.time_stretch(signal, rate=0.9)

    return signal


# =========================================================
# Feature Extraction
# =========================================================

def extract_features(file_path, augment=False):

    signal, sr = librosa.load(file_path, sr=16000)

    if augment:
        signal = augment_audio(signal, sr)

    signal, _ = librosa.effects.trim(signal)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = (mel_db - np.mean(mel_db)) / np.std(mel_db)

    if mel_db.shape[1] < 128:
        pad = 128 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)))

    else:
        mel_db = mel_db[:, :128]

    return mel_db


# =========================================================
# Build Dataset
# =========================================================

def build_dataset(file_list, label_list, augment=False):

    X = []
    y = []

    for f, l in zip(file_list, label_list):

        features = extract_features(f, augment)

        X.append(features)
        y.append(l)

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    return X, y


X_train, y_train = build_dataset(train_files, train_labels, augment=True)
X_val, y_val = build_dataset(val_files, val_labels)
X_test, y_test = build_dataset(test_files, test_labels)

print("Train shape:", X_train.shape)


# =========================================================
# CNN Model
# =========================================================

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


# =========================================================
# Compile Model
# =========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# Early Stopping
# =========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)


# =========================================================
# Train Model
# =========================================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)


# =========================================================
# Plot Training Graph
# =========================================================

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train","Validation"])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["Train","Validation"])

plt.show()


# =========================================================
# Evaluate Model
# =========================================================

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)


# =========================================================
# Predictions
# =========================================================

pred_probs = model.predict(X_test)

pred_classes = np.argmax(pred_probs, axis=1)

print("Overall Accuracy:", accuracy_score(y_test, pred_classes))


# =========================================================
# Classification Report
# =========================================================

print(classification_report(
    y_test,
    pred_classes,
    target_names=class_names
))


# =========================================================
# Confusion Matrix
# =========================================================

cm = confusion_matrix(y_test, pred_classes)

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


# =========================================================
# Save Model
# =========================================================

model.save("instrument_cnn_model.h5")

print("Model saved successfully")


# =========================================================
# Random Prediction Demo
# =========================================================

sample = np.random.randint(0, len(X_test))

prediction = model.predict(X_test[sample:sample+1])[0]

predicted_class = np.argmax(prediction)

print("\nActual Instrument:", class_names[y_test[sample]])
print("Predicted Instrument:", class_names[predicted_class])


# =========================================================
# Top 3 Predictions
# =========================================================

top3 = np.argsort(prediction)[-3:][::-1]

print("\nTop 3 Predictions:")

for i in top3:
    print(class_names[i], ":", prediction[i])