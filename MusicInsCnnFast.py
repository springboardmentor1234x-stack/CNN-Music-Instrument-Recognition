# ===============================
# 1. Import Libraries
# ===============================

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

# ===============================
# 2. Dataset Location
# ===============================

DATA_DIR = r"D:\MusicInstrumentCNN\IRMAS-TrainingData"

# ===============================
# 3. Load Dataset (LIMITED FILES)
# ===============================

audio_files = []
labels = []

for folder in os.listdir(DATA_DIR):

    folder_path = os.path.join(DATA_DIR, folder)

    if os.path.isdir(folder_path):

        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                audio_files.append(os.path.join(folder_path,file))
                labels.append(folder)

# USE SMALL DATASET FOR SPEED
audio_files = audio_files[:600]
labels = labels[:600]

print("Dataset size:", len(audio_files))

# ===============================
# 4. Encode Labels
# ===============================

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

class_names = encoder.classes_

# ===============================
# 5. Feature Extraction
# ===============================

def extract_features(file):

    signal, sr = librosa.load(file, sr=16000, duration=3)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=64
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < 64:

        pad = 64 - mel_db.shape[1]
        mel_db = np.pad(mel_db,((0,0),(0,pad)))

    else:

        mel_db = mel_db[:,:64]

    return mel_db

# ===============================
# 6. Build Feature Dataset
# ===============================

X = []

for f in audio_files:

    feature = extract_features(f)
    X.append(feature)

X = np.array(X)
X = X[...,np.newaxis]

print("Feature shape:", X.shape)

# ===============================
# 7. Train / Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,y_train,test_size=0.2,random_state=42,stratify=y_train
)

# ===============================
# 8. CNN Model
# ===============================

model = models.Sequential([

    layers.Conv2D(16,(3,3),activation="relu",input_shape=(64,64,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(64,activation="relu"),

    layers.Dense(len(class_names),activation="softmax")

])

model.summary()

# ===============================
# 9. Compile Model
# ===============================

model.compile(

    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]

)

# ===============================
# 10. Train Model
# ===============================

history = model.fit(

    X_train,
    y_train,
    epochs=5,
    batch_size=16,
    validation_data=(X_val,y_val)

)

# ===============================
# 11. Evaluation
# ===============================

loss,acc = model.evaluate(X_test,y_test)

print("Test Accuracy:",acc)

# ===============================
# 12. Predictions
# ===============================

pred = model.predict(X_test)

pred_classes = np.argmax(pred,axis=1)

print(classification_report(
    y_test,
    pred_classes,
    target_names=class_names
))

# ===============================
# 13. Confusion Matrix
# ===============================

import seaborn as sns

cm = confusion_matrix(y_test,pred_classes)

plt.figure(figsize=(8,6))

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
# 14. Training Graphs
# ===============================

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