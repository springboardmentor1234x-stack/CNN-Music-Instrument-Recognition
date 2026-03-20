

import librosa
import numpy as np
import tensorflow as tf
import cv2
import os

def save_spectrogram(file_path, save_path):

    # LOAD AUDIO
    y, sr = librosa.load(file_path, sr=22050)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    #  NORMALIZATION
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    #  DELTA FEATURES
    delta = librosa.feature.delta(mel_spec_db)
    delta2 = librosa.feature.delta(mel_spec_db, order=2)

    # STACK INTO 3 CHANNELS
    combined = np.stack([mel_spec_db, delta, delta2], axis=-1)

    # NORMALIZE
    combined = (combined - combined.min()) / (combined.max() - combined.min())

    # Resize
    combined = tf.image.resize(combined, (128,128)).numpy()

    # Convert to image
    combined = (combined * 255).astype(np.uint8)


    #  SAVE IMAGE
    cv2.imwrite(save_path, combined)
input_dataset = "/content/drive/MyDrive/IRMAS-TrainingData"
output_dataset = "/content/drive/MyDrive/spectrogram_datasets"

os.makedirs(output_dataset, exist_ok=True)

for root, dirs, files in os.walk(input_dataset):

    for file in files:
        if file.endswith(".wav"):

            file_path = os.path.join(root, file)


            class_name = os.path.basename(root)


            save_folder = os.path.join(output_dataset, class_name)
            os.makedirs(save_folder, exist_ok=True)


            save_path = os.path.join(save_folder, file.replace(".wav", ".png"))


            save_spectrogram(file_path, save_path)

print(" Dataset generated successfully!")
