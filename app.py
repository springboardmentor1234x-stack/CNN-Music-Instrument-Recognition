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
st.set_page_config(
    page_title="InstruNet AI",
    layout="centered",
    page_icon="🎵",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM UI ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    .block-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 32px;
        padding: 2rem !important;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        max-width: 1200px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    div[data-testid="stVerticalBlock"] > div:empty {
        display: none;
    }

    .main-title {
        font-size: 56px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #dbeafe 0%, #93c5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        animation: gradientShift 3s ease infinite;
        font-family: 'Inter', sans-serif;
    }

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .subtitle {
        text-align: center;
        color: #d1d5db;
        margin-bottom: 40px;
        font-size: 18px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        position: relative;
    }

    .subtitle::after {
        content: '';
        display: block;
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #60a5fa, #3b82f6);
        margin: 12px auto 0;
        border-radius: 2px;
    }

    .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 32px;
        border-radius: 24px;
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px -15px rgba(30, 60, 114, 0.4);
        transition: all 0.3s ease;
        animation: slideInUp 0.5s ease-out;
        font-family: 'Inter', sans-serif;
    }

    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px -12px rgba(30, 60, 114, 0.6);
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stFileUploader label {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }

    .stFileUploader {
        background: rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(30, 60, 114, 0.4);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
    }

    .stAudio {
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }

    h1, h2, h3, .stSubheader {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #f8fafc !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }

    h2 {
        font-size: 28px !important;
        border-left: 4px solid #60a5fa;
        padding-left: 16px;
    }

    h3 {
        font-size: 22px !important;
        color: #e5e7eb !important;
    }

    p, .stMarkdown, .stText {
        color: #d1d5db !important;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
    }

    .stCodeBlock, pre {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid rgba(255,255,255,0.12);
        color: #f8fafc !important;
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }

    code {
        color: #93c5fd !important;
        background: rgba(255,255,255,0.08) !important;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 500;
    }

    .waveform-container {
        background: rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    .footer {
        text-align: center;
        padding: 2rem;
        color: #cbd5e1;
        font-size: 12px;
        margin-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }

    .stSuccess {
        background: rgba(34, 197, 94, 0.15) !important;
        color: #dcfce7 !important;
        border-radius: 12px !important;
        border-left: 4px solid #22c55e !important;
        padding: 1rem !important;
    }

    .stInfo {
        background: rgba(59, 130, 246, 0.15) !important;
        color: #dbeafe !important;
        border-radius: 12px !important;
        border-left: 4px solid #3b82f6 !important;
    }

    [data-testid="stMetricValue"] {
        color: #dbeafe !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    .stSlider label,
    .stSelectbox label,
    .stRadio label,
    .stCheckbox label,
    .stNumberInput label,
    .stTextInput label {
        color: #f8fafc !important;
    }

    .streamlit-expanderHeader {
        color: #f8fafc !important;
        font-weight: 600 !important;
        background: rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
    }

    .stTabs [aria-selected="true"] {
        color: #dbeafe !important;
        background: rgba(255,255,255,0.12) !important;
        border-bottom: none !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    @media (max-width: 768px) {
        .main-title {
            font-size: 36px;
        }

        .result-card {
            font-size: 24px;
            padding: 20px;
        }

        .block-container {
            padding: 1rem !important;
        }

        h2 {
            font-size: 24px !important;
        }

        h3 {
            font-size: 20px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER WITH ANIMATION ----------------
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div class="main-title">🎵 InstruNet AI</div>
    <div class="subtitle">Automatic Musical Instrument Detection from Audio</div>
</div>
""", unsafe_allow_html=True)

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
    lines = []
    for inst, val in scores.items():
        bars = "█" * int(val * 20)
        lines.append(f"{inst:20} | {bars:<20} | {val:.2f}")
    return "\n".join(lines)

# ---------------- WAVEFORM IMAGE ----------------
def create_waveform_image(audio_path):
    y, sr = librosa.load(audio_path, mono=True)

    plt.figure(figsize=(8, 3))
    plt.style.use("default")
    plt.plot(y, color="#2a5298", alpha=0.8, linewidth=1)
    plt.title("Audio Waveform Analysis", color="#1a202c", fontsize=14, fontweight="bold")
    plt.xlabel("Time (seconds)", color="#4a5568")
    plt.ylabel("Amplitude", color="#4a5568")
    plt.grid(alpha=0.3, color="#cbd5e0")
    plt.tight_layout()
    plt.savefig("waveform.png", facecolor="white", edgecolor="none", dpi=100)
    plt.close()

    return "waveform.png"

# ---------------- CONFIDENCE GRAPH ----------------
def create_confidence_graph(scores):
    names = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(6, 3))
    plt.style.use("default")
    colors = ["#1e3c72", "#2a5298", "#3b6cb0", "#4c7fc9"]
    bars = plt.bar(names, values, color=colors, alpha=0.8)
    plt.ylabel("Confidence Score", color="#4a5568", fontsize=12)
    plt.ylim(0, 1)
    plt.title("Instrument Detection Confidence", color="#1a202c", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, color="#cbd5e0")
    plt.tight_layout()

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            color="#1a202c",
            fontweight="bold",
        )

    plt.savefig("confidence.png", facecolor="white", edgecolor="none", dpi=100)
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
    elements.append(Paragraph(f"<b>Audio File:</b> {result['audio_file']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Detected Instrument:</b> {result['detected_instrument']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Confidence:</b> {result['confidence']:.2f}", styles["Normal"]))

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

# ---------------- UPLOAD SECTION ----------------
uploaded_file = st.file_uploader(
    "🎵 Choose your audio file (WAV or MP3)",
    type=["wav", "mp3"],
    help="Upload a clear recording of a musical instrument for best results"
)

if uploaded_file is not None:
    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🎧 Audio Preview")
        st.audio("input_audio.wav")

    with st.spinner("🎵 Analyzing your audio with InstruNet AI..."):
        X_test = audio_to_spectrogram("input_audio.wav")
        pred = model.predict(X_test)[0]

        detected_code = labels[np.argmax(pred)]
        detected_name, icon = label_map[detected_code]
        confidence = float(np.max(pred))

        waveform_path = create_waveform_image("input_audio.wav")

    st.markdown("### 📊 Audio Analysis")
    st.markdown('<div class="waveform-container">', unsafe_allow_html=True)
    st.image(waveform_path, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="result-card">{icon} {detected_name}<br><span style="font-size: 18px;">Confidence: {confidence:.1%}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📈 Confidence Analysis")
    chart_data = {
        label_map[labels[i]][0]: float(pred[i]) for i in range(len(pred))
    }
    st.bar_chart(chart_data)

    confidence_path = create_confidence_graph(chart_data)
    st.image(confidence_path, use_container_width=True)

    # 🎯 Instrument Intensity (Dashboard Style + PDF text source)
    st.markdown("### 🎯 Instrument Intensity")
    intensity_text = generate_intensity_text(chart_data)

    for inst, val in chart_data.items():
        st.progress(val, text=f"{inst} ({val:.2f})")

    result = {
        "audio_file": uploaded_file.name,
        "detected_instrument": detected_name,
        "confidence": confidence,
        "scores": chart_data,
    }

    json_str = json.dumps(result, indent=4)

    st.markdown("### 📥 Download Reports")
    st.markdown("Get detailed analysis reports in your preferred format")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📄 Download JSON Report",
            json_str,
            file_name="prediction.json",
            mime="application/json",
            use_container_width=True,
        )

    pdf_path = generate_pdf(result, waveform_path, confidence_path, intensity_text)

    with open(pdf_path, "rb") as f:
        with col2:
            st.download_button(
                "📑 Download PDF Report",
                f,
                file_name="prediction.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    st.success("✅ Analysis complete! Your reports are ready for download.")

st.markdown("""
<div class="footer">
    <p style="color: #cbd5e1;">InstruNet AI - Powered by Deep Learning | Made with 🎵 for musicians and audio enthusiasts</p>
    <p style="color: #94a3b8; font-size: 10px;">Supports Piano, Acoustic Guitar, Electric Guitar, and Violin detection</p>
</div>
""", unsafe_allow_html=True)
