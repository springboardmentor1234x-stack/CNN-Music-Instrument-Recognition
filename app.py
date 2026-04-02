import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import gdown

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Preformatted
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="InstruNet AI", layout="centered")

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🎵 InstruNet AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Automatic Musical Instrument Detection</p>", unsafe_allow_html=True)

# =========================================================
# 🔥 SIDEBAR (MUST BE HERE - TOP LEVEL)
# =========================================================
st.sidebar.title("⚙️ Hyperparameter Tuning")

st.sidebar.write("Adjust preprocessing parameters 👇")

n_fft = st.sidebar.slider("FFT Size", 512, 4096, 2048, step=512)
hop_length = st.sidebar.slider("Hop Length", 128, 1024, 256, step=128)
n_mels = st.sidebar.slider("Mel Bands", 64, 256, 128, step=32)

colormap = st.sidebar.selectbox(
    "Spectrogram Color",
    ["magma", "viridis", "plasma"]
)

threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# ---------------- LABELS ----------------
label_map = {
    "pia": ("Piano", "🎹"),
    "gac": ("Acoustic Guitar", "🎸"),
    "gel": ("Electric Guitar", "⚡🎸"),
    "vio": ("Violin", "🎻")
}
labels = list(label_map.keys())

# ---------------- MODEL ----------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "instrunet_cnn.keras")

FILE_ID = "1qVlfOXIVthbxdYFQfrxsxCSo1sJTMrXb"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=True, fuzzy=True)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- AUDIO → SPECTROGRAM ----------------
def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mel_db, cmap=colormap)
    plt.axis("off")
    plt.savefig("temp_spec.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open("temp_spec.png").convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0

    return np.expand_dims(img, axis=0)

# ---------------- WAVEFORM ----------------
def create_waveform_image(audio_path):
    y, sr = librosa.load(audio_path, mono=True)

    plt.figure(figsize=(8, 3))
    librosa.display.waveshow(y, sr=sr, color="cyan")
    plt.title("Audio Waveform")
    plt.tight_layout()
    plt.savefig("waveform.png")
    plt.close()

    return "waveform.png"

# ---------------- CONFIDENCE GRAPH ----------------
def create_confidence_graph(scores):
    names = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(6, 3))
    plt.bar(names, values)
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("confidence.png")
    plt.close()

    return "confidence.png"

# ---------------- INTENSITY ----------------
def generate_intensity_text(scores):
    text = "Instrument Intensity:\n"
    for inst, val in scores.items():
        bars = "|" * int(val * 20)
        text += f"{inst}: {bars}\n"
    return text

# ---------------- PDF ----------------
def generate_pdf(result, waveform_path, confidence_path, intensity_text):
    pdf_path = "report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("InstruNet AI Report", styles["Title"]))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph(f"Audio File: {result['audio_file']}", styles["Normal"]))
    elements.append(Paragraph(f"Detected Instrument: {result['detected_instrument']}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {result['confidence']:.2f}", styles["Normal"]))

    elements.append(Spacer(1, 10))
    elements.append(Preformatted(intensity_text, styles["Code"]))

    if os.path.exists(waveform_path):
        elements.append(RLImage(waveform_path, width=400, height=150))

    if os.path.exists(confidence_path):
        elements.append(RLImage(confidence_path, width=400, height=200))

    doc.build(elements)
    return pdf_path

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Audio (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:

    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("input_audio.wav")

    with st.spinner("Analyzing..."):
        X_test = audio_to_spectrogram("input_audio.wav")
        pred = model.predict(X_test)[0]

    idx = np.argmax(pred)
    detected_name, icon = label_map[labels[idx]]
    confidence = float(np.max(pred))

    # 🔥 Threshold logic
    if confidence < threshold:
        detected_name = "Uncertain"
        icon = "⚠️"

    waveform_path = create_waveform_image("input_audio.wav")

    st.subheader("Audio Visualization")
    st.image(waveform_path)

    st.markdown(f"## {icon} {detected_name} ({confidence:.2f})")

    chart_data = {
        label_map[labels[i]][0]: float(pred[i])
        for i in range(len(pred))
    }

    st.bar_chart(chart_data)

    confidence_path = create_confidence_graph(chart_data)

    intensity_text = generate_intensity_text(chart_data)

    st.subheader("Instrument Intensity")
    st.code(intensity_text)

    result = {
        "audio_file": uploaded_file.name,
        "detected_instrument": detected_name,
        "confidence": confidence,
        "scores": chart_data
    }

    st.download_button("Download JSON", json.dumps(result, indent=4))

    pdf_path = generate_pdf(result, waveform_path, confidence_path, intensity_text)

    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF", f)
