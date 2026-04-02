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
st.set_page_config(
    page_title="InstruNet AI",
    layout="centered",
    page_icon="🎵"
)

# ---------------- PREMIUM UI ----------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.block-container {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
    border-radius: 28px;
    padding: 2rem !important;
    border: 1px solid rgba(255,255,255,0.08);
}

.main-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg,#dbeafe,#93c5fd);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle {
    text-align:center;
    color:#d1d5db;
    margin-bottom:30px;
}

.result-card {
    background: linear-gradient(135deg,#1e3c72,#2a5298);
    padding:30px;
    border-radius:20px;
    text-align:center;
    font-size:30px;
    font-weight:800;
    color:white;
}

.footer {
    text-align:center;
    margin-top:40px;
    color:#94a3b8;
}
</style>""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🎵 InstruNet AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Automatic Musical Instrument Detection from Audio</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Hyperparameter Tuning")

n_fft = st.sidebar.slider("FFT Size", 512, 4096, 2048, step=512)
hop_length = st.sidebar.slider("Hop Length", 128, 1024, 256, step=128)
n_mels = st.sidebar.slider("Mel Bands", 64, 256, 128, step=32)

colormap = st.sidebar.selectbox("Color Map", ["magma", "viridis", "plasma"])
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# ---------------- LABELS ----------------
label_map = {
    "pia": ("Piano", "🎹"),
    "gac": ("Acoustic Guitar", "🎸"),
    "gel": ("Electric Guitar", "⚡🎸"),
    "vio": ("Violin", "🎻"),
}
labels = list(label_map.keys())

# ---------------- MODEL ----------------
MODEL_PATH = "model/instrunet_cnn.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- AUDIO → SPECTROGRAM ----------------
def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(4,4))
    librosa.display.specshow(mel_db, cmap=colormap)
    plt.axis("off")
    plt.savefig("temp.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open("temp.png").convert("RGB").resize((224,224))
    return np.expand_dims(np.array(img)/255.0, axis=0)

# ---------------- FUNCTIONS ----------------
def create_waveform(audio):
    y, sr = librosa.load(audio)
    plt.figure(figsize=(8,3))
    librosa.display.waveshow(y, sr=sr)
    plt.tight_layout()
    plt.savefig("wave.png")
    plt.close()
    return "wave.png"

def create_graph(scores):
    plt.figure(figsize=(6,3))
    plt.bar(list(scores.keys()), list(scores.values()))
    plt.ylim(0,1)
    plt.savefig("conf.png")
    plt.close()
    return "conf.png"

def generate_intensity(scores):
    return "\n".join([f"{k}: {'|'*int(v*20)}" for k,v in scores.items()])

# ---------------- UPLOAD ----------------
file = st.file_uploader("🎧 Upload Audio (.wav/.mp3)", type=["wav","mp3"])

if file:
    with open("input.wav","wb") as f:
        f.write(file.read())

    st.audio("input.wav")

    with st.spinner("Analyzing..."):
        pred = model.predict(audio_to_spectrogram("input.wav"))[0]

    idx = np.argmax(pred)
    name, icon = label_map[labels[idx]]
    conf = float(np.max(pred))

    if conf < threshold:
        name, icon = "Uncertain","⚠️"

    st.markdown(f'<div class="result-card">{icon} {name} ({conf:.2f})</div>', unsafe_allow_html=True)

    wave = create_waveform("input.wav")
    st.image(wave)

    scores = {label_map[labels[i]][0]:float(pred[i]) for i in range(len(pred))}
    st.bar_chart(scores)

    intensity = generate_intensity(scores)
    st.code(intensity)

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">InstruNet AI • Deep Learning Project</div>', unsafe_allow_html=True)
