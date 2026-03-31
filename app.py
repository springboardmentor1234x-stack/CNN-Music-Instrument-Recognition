# ============================================
# CNN-Based Music Instrument Recognition System
# Flask App - Backend
# ============================================

import os
import io
import json
import uuid
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ============================================
# FLASK CONFIG
# ============================================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PLOT_FOLDER   = "static/plots"
MODEL_PATH    = "models/instrunet_cnn.h5"
LABELS_PATH   = "models/label_classes.json"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PLOT_FOLDER"]   = PLOT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac"}

# ============================================
# AUDIO / MODEL CONFIG
# ============================================
SAMPLE_RATE  = 22050
DURATION     = 3
N_MELS       = 128
HOP_LENGTH   = 512
IMG_SIZE     = (128, 128)
SEGMENT_SECS = 3
THRESHOLD    = 0.30

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

# ============================================
# LOAD MODEL
# ============================================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        classes = json.load(f)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None
    classes = list(INSTRUMENT_NAMES.keys())

# ============================================
# HELPERS
# ============================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_audio(file_path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio

def audio_to_mel(audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop=HOP_LENGTH):
    target = sr * DURATION
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def mel_to_input(mel_db):
    fig, ax = plt.subplots(figsize=(2, 2), dpi=64)
    librosa.display.specshow(mel_db, ax=ax, cmap="magma")
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB").resize(IMG_SIZE)
    return img_to_array(img) / 255.0

def predict_instruments(audio, sr=SAMPLE_RATE):
    segment_len = sr * SEGMENT_SECS
    segments = [
        audio[i:i + segment_len]
        for i in range(0, len(audio), segment_len)
        if len(audio[i:i + segment_len]) >= sr
    ]

    if len(segments) == 0:
        segments = [audio]

    all_preds = []
    for seg in segments:
        mel = audio_to_mel(seg)
        x = mel_to_input(mel)[np.newaxis, ...]
        probs = model.predict(x, verbose=0)[0]
        all_preds.append(probs)

    timeline = np.array(all_preds)
    avg_pred = np.mean(timeline, axis=0)
    return avg_pred, timeline

# ============================================
# PLOT FUNCTIONS
# ============================================
def save_waveform(audio, filename, sr=SAMPLE_RATE):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax, color="#1f77b4")
    ax.set_title("Waveform")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()

    path = os.path.join(PLOT_FOLDER, filename)
    plt.savefig(path)
    plt.close(fig)
    return filename

def save_spectrogram(audio, filename, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-Spectrogram")
    plt.tight_layout()

    path = os.path.join(PLOT_FOLDER, filename)
    plt.savefig(path)
    plt.close(fig)
    return filename

def save_timeline_plot(timeline, filename):
    readable = [INSTRUMENT_NAMES.get(c, c) for c in classes]
    n_seg = timeline.shape[0]
    x = np.arange(n_seg) * SEGMENT_SECS

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))

    for i, (name, color) in enumerate(zip(readable, colors)):
        ax.plot(x, timeline[:, i] * 100, label=name, color=color, linewidth=2)

    ax.axhline(y=THRESHOLD * 100, color="gold", linestyle="--", linewidth=1, label="Threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Instrument Activity Over Time")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()

    path = os.path.join(PLOT_FOLDER, filename)
    plt.savefig(path)
    plt.close(fig)
    return filename

# ============================================
# ROUTES
# ============================================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        if "audio_file" not in request.files:
            return render_template("index.html", result=result, error="No file uploaded.")

        file = request.files["audio_file"]

        if file.filename == "":
            return render_template("index.html", result=result, error="Please select an audio file.")

        if file and allowed_file(file.filename):
            unique_id = str(uuid.uuid4())[:8]
            filename = secure_filename(file.filename)
            saved_name = f"{unique_id}_{filename}"
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
            file.save(upload_path)

            # Load audio
            audio = load_audio(upload_path)
            duration = len(audio) / SAMPLE_RATE

            # Save plots
            waveform_file = save_waveform(audio, f"{unique_id}_waveform.png")
            spectrogram_file = save_spectrogram(audio, f"{unique_id}_spectrogram.png")

            # Prediction
            if model is not None:
                avg_pred, timeline = predict_instruments(audio)
                timeline_file = save_timeline_plot(timeline, f"{unique_id}_timeline.png")

                predictions = []
                for i, c in enumerate(classes):
                    confidence = round(float(avg_pred[i]) * 100, 2)
                    present = bool(avg_pred[i] >= THRESHOLD)

                    predictions.append({
                        "code": c,
                        "name": INSTRUMENT_NAMES.get(c, c),
                        "confidence": confidence,
                        "present": present
                    })

                # Sort by highest confidence
                predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

                # Advanced grouped outputs
                top_3_predictions = predictions[:3]
                present_instruments = [p for p in predictions if p["present"]]
                absent_instruments = [p for p in predictions if not p["present"]]

            else:
                timeline_file = None
                predictions = []
                top_3_predictions = []
                present_instruments = []
                absent_instruments = []

            result = {
                "filename": filename,
                "audio_url": url_for("static", filename=f"uploads/{saved_name}"),
                "waveform_url": url_for("static", filename=f"plots/{waveform_file}"),
                "spectrogram_url": url_for("static", filename=f"plots/{spectrogram_file}"),
                "timeline_url": url_for("static", filename=f"plots/{timeline_file}") if timeline_file else None,
                "duration": round(duration, 2),
                "sample_rate": SAMPLE_RATE,
                "predictions": predictions,
                "top_3_predictions": top_3_predictions,
                "present_instruments": present_instruments,
                "absent_instruments": absent_instruments
            }

            return render_template("index.html", result=result)

        else:
            return render_template("index.html", error="Invalid file type. Upload WAV, MP3, or FLAC.")

    return render_template("index.html", result=result)
# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    app.run(debug=True)