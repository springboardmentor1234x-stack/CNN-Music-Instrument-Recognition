import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Dataset Path
# -----------------------------

DATASET_PATH = "D:\MusicInstrumentCNN\IRMAS-TrainingData"

paths = []
labels = []

for inst in os.listdir(DATASET_PATH):

    inst_path = os.path.join(DATASET_PATH, inst)

    if os.path.isdir(inst_path):

        for f in os.listdir(inst_path):

            if f.endswith(".wav"):

                paths.append(os.path.join(inst_path,f))
                labels.append(inst)

print("Dataset size:",len(paths))


# -----------------------------
# Encode Labels
# -----------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
class_names = encoder.classes_

# -----------------------------
# Split Dataset
# -----------------------------

train_files,temp_files,y_train,y_temp = train_test_split(
    paths,y,test_size=0.2,stratify=y,random_state=42)

val_files,test_files,y_val,y_test = train_test_split(
    temp_files,y_temp,test_size=0.5,stratify=y_temp,random_state=42)

print("Train:",len(train_files)," Val:",len(val_files)," Test:",len(test_files))


# -----------------------------
# Feature Extraction
# -----------------------------

def extract(file):

    audio,sr = librosa.load(file,sr=16000)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=64
    )

    mel_db = librosa.power_to_db(mel,ref=np.max)

    if mel_db.shape[1] < 64:
        pad = 64-mel_db.shape[1]
        mel_db = np.pad(mel_db,((0,0),(0,pad)))

    else:
        mel_db = mel_db[:,:64]

    return mel_db


def build(files,labels):

    X=[]
    y=[]

    for f,l in zip(files,labels):

        X.append(extract(f))
        y.append(l)

    X=np.array(X)[...,np.newaxis]
    y=np.array(y)

    return X,y


print("Processing dataset...")

X_train,y_train = build(train_files,y_train)
X_val,y_val = build(val_files,y_val)
X_test,y_test = build(test_files,y_test)

print("Train shape:",X_train.shape)


# -----------------------------
# CNN Model
# -----------------------------

model = models.Sequential([

    layers.Input(shape=(64,64,1)),

    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128,activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(len(class_names),activation="softmax")

])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Train Model
# -----------------------------

early_stop = EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=8,
    batch_size=64,
    validation_data=(X_val,y_val),
    callbacks=[early_stop]
)


# -----------------------------
# Evaluate
# -----------------------------

loss,acc = model.evaluate(X_test,y_test)

print("Test Accuracy:",acc)


# -----------------------------
# Predictions
# -----------------------------

pred = model.predict(X_test)

pred_classes = np.argmax(pred,axis=1)

print("Overall Accuracy:",accuracy_score(y_test,pred_classes))

print(classification_report(y_test,pred_classes,target_names=class_names))


# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test,pred_classes)

plt.figure(figsize=(8,6))

sns.heatmap(cm,annot=True,fmt="d",
xticklabels=class_names,
yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# -----------------------------
# Save Model
# -----------------------------

model.save("instrument_model.h5")

print("Model saved")


# -----------------------------
# Demo Prediction
# -----------------------------

i = np.random.randint(0,len(X_test))

p = model.predict(X_test[i:i+1])[0]

print("\nActual:",class_names[y_test[i]])
print("Predicted:",class_names[np.argmax(p)])