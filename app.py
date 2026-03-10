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
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Preformatted,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="InstruNet AI", layout="centered")

# ---------------- CUSTOM UI ----------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 4px;
    }
    .subtitle {
        text-align: center;
        color: #9aa0a6;
        margin-bottom: 30px;
        font-size: 16px;
    }
    .result-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        color: white;
        padding: 22px;
        border-radius: 16px;
        text-align: center;
        font-size: 26px;
        font-weight: 700;
        margin-top: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">InstruNet AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Automatic Musical Instrument Detection from Audio</div>',
    unsafe_allow_html=True,
)

# ---------------- LABELS + ICONS ----------------
label_map = {
    "pia": ("Piano", "🎹"),
    "gac": ("Acoustic Guitar", "🎸"),
    "gel": ("Electric Guitar", "⚡🎸"),
    "vio": ("Violin", "🎻"),
}
labels = list(label_map.keys())

# ---------------- MODEL SETUP ----------------
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
def audio_to_spectrogram(audio_path, img_size=224):
    y, sr = librosa.load(audio_path, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=4096,
        hop_length=256,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mel_db, cmap="magma")
    plt.axis("off")
    plt.savefig("temp_spec.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open("temp_spec.png").convert("RGB")
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0

    return np.expand_dims(img, axis=0)


# ---------------- INTENSITY TEXT ----------------
def generate_intensity_text(scores):
    text = "Instrument Intensity:\n"
    for inst, val in scores.items():
        bars = "|" * int(val * 20)
        text += f"{inst}: {bars}\n"
    return text


# ---------------- WAVEFORM IMAGE ----------------
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


# ---------------- PDF GENERATION ----------------
def generate_pdf(result, waveform_path, confidence_path, intensity_text):
    pdf_path = "report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("InstruNet AI Report", styles["Title"]))
    elements.append(Spacer(1, 15))
    elements.append(
        Paragraph(f"<b>Audio File:</b> {result['audio_file']}", styles["Normal"])
    )
    elements.append(
        Paragraph(
            f"<b>Detected Instrument:</b> {result['detected_instrument']}",
            styles["Normal"],
        )
    )
    elements.append(
        Paragraph(f"<b>Confidence:</b> {result['confidence']:.2f}", styles["Normal"])
    )

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Instrument Intensity:</b>", styles["Heading2"]))
    elements.append(Preformatted(intensity_text, styles["Code"]))
    elements.append(Spacer(1, 15))

    if os.path.exists(waveform_path):
        elements.append(Paragraph("<b>Audio Waveform:</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        elements.append(RLImage(waveform_path, width=400, height=150))
        elements.append(Spacer(1, 15))

    if os.path.exists(confidence_path):
        elements.append(Paragraph("<b>Confidence Scores:</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        elements.append(RLImage(confidence_path, width=400, height=200))

    doc.build(elements)
    return pdf_path


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Choose a .wav or .mp3 file",
    type=["wav", "mp3"],
)

if uploaded_file is not None:
    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("input_audio.wav")

    with st.spinner("Analyzing audio..."):
        X_test = audio_to_spectrogram("input_audio.wav")
        pred = model.predict(X_test)[0]

        detected_code = labels[np.argmax(pred)]
        detected_name, icon = label_map[detected_code]
        confidence = float(np.max(pred))

        waveform_path = create_waveform_image("input_audio.wav")

    st.subheader("Audio Visualization")
    st.image(waveform_path, use_container_width=True)

    st.markdown(
        f'<div class="result-card">'
        f'{icon} Detected Instrument: {detected_name} ({confidence:.2f})'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Confidence Scores")
    chart_data = {
        label_map[labels[i]][0]: float(pred[i]) for i in range(len(pred))
    }
    st.bar_chart(chart_data)

    confidence_path = create_confidence_graph(chart_data)

    # ---------------- INTENSITY TEXT ----------------
    intensity_text = generate_intensity_text(chart_data)
    st.subheader("Instrument Intensity")
    st.code(intensity_text)

    result = {
        "audio_file": uploaded_file.name,
        "detected_instrument": detected_name,
        "confidence": confidence,
        "scores": chart_data,
    }

    json_str = json.dumps(result, indent=4)

    st.subheader("Download Report")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download JSON Report",
            json_str,
            file_name="prediction.json",
            mime="application/json",
            use_container_width=True,
        )

    pdf_path = generate_pdf(
        result,
        waveform_path,
        confidence_path,
        intensity_text,
    )

    with open(pdf_path, "rb") as f:
        with col2:
            st.download_button(
                "Download PDF Report",
                f,
                file_name="prediction.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
