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
import sqlite3
import hashlib
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Preformatted
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- DATABASE ----------------
DB_PATH = "users.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(full_name, email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (full_name, email, password, created_at) VALUES (?, ?, ?, ?)",
            (full_name, email, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        return True, "Registration successful. Please sign in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    except Exception as e:
        return False, f"Registration failed: {e}"


def login_user(email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, full_name, email FROM users WHERE email = ? AND password = ?",
            (email, hash_password(password))
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            return True, {
                "id": user[0],
                "full_name": user[1],
                "email": user[2]
            }
        return False, "Invalid email or password."
    except Exception as e:
        return False, f"Login failed: {e}"


init_db()

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "landing"

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #081120 0%, #0f172a 45%, #111827 100%);
        color: #f8fafc;
    }

    .main > div {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #101827 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(37,99,235,0.16), rgba(99,102,241,0.13), rgba(14,165,233,0.12));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 28px;
        padding: 42px 38px;
        backdrop-filter: blur(14px);
        box-shadow: 0 14px 40px rgba(0,0,0,0.28);
        margin-bottom: 22px;
    }

    .hero-card::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(59,130,246,0.28) 0%, rgba(59,130,246,0.0) 70%);
        border-radius: 50%;
    }

    .hero-badge {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        color: #dbeafe;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.08;
        color: #ffffff;
        margin-bottom: 0.65rem;
        letter-spacing: -0.6px;
        max-width: 850px;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: #dbeafe;
        max-width: 820px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .info-chip-wrap {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 22px;
    }

    .info-chip {
        background: rgba(255,255,255,0.07);
        color: #eff6ff;
        padding: 9px 15px;
        border-radius: 999px;
        font-size: 0.92rem;
        border: 1px solid rgba(255,255,255,0.10);
    }

    .landing-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }

    .landing-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        min-height: 180px;
    }

    .landing-icon {
        font-size: 1.9rem;
        margin-bottom: 10px;
    }

    .landing-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .landing-text {
        color: #cbd5e1;
        font-size: 0.96rem;
        line-height: 1.7;
    }

    .section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 12px;
        color: #ffffff;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(59,130,246,0.12));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .metric-label {
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    .upload-box {
        background: rgba(255,255,255,0.03);
        border: 1.5px dashed rgba(148,163,184,0.4);
        border-radius: 20px;
        padding: 18px;
    }

    .result-banner {
        background: linear-gradient(135deg, rgba(14,165,233,0.16), rgba(99,102,241,0.18));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 18px 20px;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .result-main {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .result-sub {
        color: #dbeafe;
        font-size: 1rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .footer {
        margin-top: 32px;
        padding: 22px 24px;
        border-radius: 20px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.8;
    }

    .footer-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .footer-sub {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 8px;
    }

    .auth-wrapper {
        max-width: 460px;
        margin: 30px auto;
    }

    .auth-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.22);
    }

    .auth-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
        text-align: center;
    }

    .auth-subtitle {
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 20px;
    }

    div[data-testid="stFileUploader"] {
        background: transparent !important;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }

    .stButton > button, .stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: white;
        font-weight: 700;
        padding: 0.72rem 1rem;
        box-shadow: 0 6px 18px rgba(37,99,235,0.25);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(255,255,255,0.20);
        background: linear-gradient(135deg, #1d4ed8, #4338ca);
        color: white;
    }

    div[data-testid="stAudio"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 10px 12px;
    }

    @media (max-width: 900px) {
        .landing-grid {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: 2.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

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
def audio_to_spectrogram(audio_path, n_mels, n_fft, hop_length, colormap):
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


def create_waveform_image(audio_path):
    y, sr = librosa.load(audio_path, mono=True)

    plt.figure(figsize=(8, 3))
    librosa.display.waveshow(y, sr=sr, color="cyan")
    plt.title("Audio Waveform")
    plt.tight_layout()
    plt.savefig("waveform.png", bbox_inches="tight")
    plt.close()

    return "waveform.png"


def create_confidence_graph(scores):
    names = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(6, 3))
    plt.bar(names, values)
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("confidence.png", bbox_inches="tight")
    plt.close()

    return "confidence.png"


def generate_intensity_text(scores):
    text = "Instrument Intensity:\n"
    for inst, val in scores.items():
        bars = "|" * int(val * 20)
        text += f"{inst}: {bars}\n"
    return text


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


# ---------------- LANDING PAGE ----------------
def show_landing_page():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-badge">AI-Powered Musical Instrument Recognition</div>
        <div class="hero-title">InstruNet AI — Professional Audio Classification Dashboard</div>
        <p class="hero-subtitle">
            A modern deep learning application that identifies musical instruments from uploaded audio,
            visualizes waveform patterns, analyzes confidence distribution, and generates exportable reports.
            Create an account or sign in to access the analysis dashboard.
        </p>
        <div class="info-chip-wrap">
            <div class="info-chip">🎵 Audio Classification</div>
            <div class="info-chip">📊 Confidence Analysis</div>
            <div class="info-chip">📈 Waveform Visualization</div>
            <div class="info-chip">📄 PDF Reporting</div>
            <div class="info-chip">🔐 Secure Access</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-grid">
        <div class="landing-card">
            <div class="landing-icon">🎼</div>
            <div class="landing-title">Smart Instrument Detection</div>
            <div class="landing-text">
                Upload an audio file and let the trained CNN model identify the most probable musical instrument
                using spectrogram-based feature extraction.
            </div>
        </div>
        <div class="landing-card">
            <div class="landing-icon">📉</div>
            <div class="landing-title">Clear Audio Insights</div>
            <div class="landing-text">
                View waveform plots, confidence scores, and intensity information in a polished interface
                suitable for demonstrations and project presentations.
            </div>
        </div>
        <div class="landing-card">
            <div class="landing-icon">📦</div>
            <div class="landing-title">Report & Export Ready</div>
            <div class="landing-text">
                Export prediction results as JSON and generate a PDF report for documentation,
                review, submission, or portfolio showcasing.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🏠 Home"):
            st.session_state.auth_mode = "landing"

    with col2:
        if st.button("📝 Register"):
            st.session_state.auth_mode = "register"

    with col3:
        if st.button("🔐 Sign In"):
            st.session_state.auth_mode = "login"

    st.markdown("""
    <div class="footer">
        <div class="footer-title">InstruNet AI</div>
        <div>
            A professional deep learning project for automatic musical instrument detection using audio signal analysis,
            spectrogram-based preprocessing, and CNN-powered classification.
        </div>
        <div class="footer-sub">
            Built with Streamlit, TensorFlow, Librosa, Matplotlib, SQLite, and ReportLab
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- REGISTER PAGE ----------------
def show_register_page():
    st.markdown('<div class="auth-wrapper"><div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Create Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Register to access the InstruNet AI dashboard</div>', unsafe_allow_html=True)

    with st.form("register_form"):
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if not full_name.strip():
                st.error("Full name is required.")
            elif not email.strip():
                st.error("Email is required.")
            elif not password:
                st.error("Password is required.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(full_name.strip(), email.strip(), password)
                if success:
                    st.success(message)
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    st.error(message)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home", key="back_home_from_register"):
            st.session_state.auth_mode = "landing"
            st.rerun()
    with col2:
        if st.button("Already have an account? Sign In", key="go_login_from_register"):
            st.session_state.auth_mode = "login"
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)


# ---------------- LOGIN PAGE ----------------
def show_login_page():
    st.markdown('<div class="auth-wrapper"><div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Sign In</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Login to continue to the InstruNet AI dashboard</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")

        if submitted:
            if not email.strip():
                st.error("Email is required.")
            elif not password:
                st.error("Password is required.")
            else:
                success, result = login_user(email.strip(), password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user = result
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error(result)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home", key="back_home_from_login"):
            st.session_state.auth_mode = "landing"
            st.rerun()
    with col2:
        if st.button("Create New Account", key="go_register_from_login"):
            st.session_state.auth_mode = "register"
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)


# ---------------- MAIN APP ----------------
def show_main_app():
    st.sidebar.markdown("## ⚙️ Hyperparameter Tuning")
    st.sidebar.markdown("Fine-tune preprocessing parameters for spectrogram generation.")

    n_fft = st.sidebar.slider("FFT Size", 512, 4096, 2048, step=512)
    hop_length = st.sidebar.slider("Hop Length", 128, 1024, 256, step=128)
    n_mels = st.sidebar.slider("Mel Bands", 64, 256, 128, step=32)

    colormap = st.sidebar.selectbox(
        "Spectrogram Color",
        ["magma", "viridis", "plasma"]
    )

    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.success(f"Logged in as {st.session_state.user['full_name']}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.auth_mode = "landing"
        st.rerun()

    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-badge">Welcome, {st.session_state.user['full_name']}</div>
        <div class="hero-title">InstruNet AI Dashboard</div>
        <p class="hero-subtitle">
            Upload an audio file to detect the musical instrument, inspect waveform and confidence output,
            and download the generated reports.
        </p>
        <div class="info-chip-wrap">
            <div class="info-chip">🎵 Audio Classification</div>
            <div class="info-chip">📊 Confidence Analysis</div>
            <div class="info-chip">📈 Waveform Visualization</div>
            <div class="info-chip">📄 PDF Reporting</div>
            <div class="info-chip">👤 {st.session_state.user['email']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Audio File</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Audio (.wav or .mp3)", type=["wav", "mp3"])
    st.markdown('<div class="small-note">Supported formats: WAV, MP3</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        with open("input_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        col_a, col_b = st.columns([1.4, 1])

        with col_a:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Uploaded Audio</div>', unsafe_allow_html=True)
            st.audio("input_audio.wav")
            st.markdown(f"<div class='small-note'>File name: {uploaded_file.name}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Model Status</div>', unsafe_allow_html=True)
            st.success("Model loaded successfully")
            st.markdown(f"**FFT Size:** {n_fft}")
            st.markdown(f"**Hop Length:** {hop_length}")
            st.markdown(f"**Mel Bands:** {n_mels}")
            st.markdown(f"**Threshold:** {threshold:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("Analyzing audio and generating prediction..."):
            X_test = audio_to_spectrogram("input_audio.wav", n_mels, n_fft, hop_length, colormap)
            pred = model.predict(X_test)[0]

        idx = np.argmax(pred)
        detected_name, icon = label_map[labels[idx]]
        confidence = float(np.max(pred))

        if confidence < threshold:
            detected_name = "Uncertain"
            icon = "⚠️"

        waveform_path = create_waveform_image("input_audio.wav")

        chart_data = {
            label_map[labels[i]][0]: float(pred[i])
            for i in range(len(pred))
        }

        confidence_path = create_confidence_graph(chart_data)
        intensity_text = generate_intensity_text(chart_data)

        st.markdown("""
        <div class="section-card">
            <div class="section-title">Prediction Summary</div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{icon}</div>
                <div class="metric-label">Detected Class</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{detected_name}</div>
                <div class="metric-label">Instrument</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence:.2f}</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="result-banner">
                <div class="result-main">{icon} {detected_name}</div>
                <div class="result-sub">Prediction confidence: {confidence:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        left_col, right_col = st.columns([1.15, 1])

        with left_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Audio Visualization</div>', unsafe_allow_html=True)
            st.image(waveform_path, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Confidence Distribution</div>', unsafe_allow_html=True)
            st.bar_chart(chart_data)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Instrument Intensity</div>', unsafe_allow_html=True)
        st.code(intensity_text)
        st.markdown('</div>', unsafe_allow_html=True)

        result = {
            "audio_file": uploaded_file.name,
            "detected_instrument": detected_name,
            "confidence": confidence,
            "scores": chart_data
        }

        pdf_path = generate_pdf(result, waveform_path, confidence_path, intensity_text)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Download Results</div>', unsafe_allow_html=True)

        dl1, dl2 = st.columns(2)

        with dl1:
            st.download_button(
                "⬇ Download JSON",
                json.dumps(result, indent=4),
                file_name="instrument_result.json",
                mime="application/json"
            )

        with dl2:
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇ Download PDF",
                    f,
                    file_name="instrument_report.pdf",
                    mime="application/pdf"
                )

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Getting Started</div>
            <p style="color:#cbd5e1; margin-bottom:0; line-height:1.8;">
                Upload an audio file to begin analysis. The system will preprocess the signal, convert it into a mel spectrogram,
                run prediction through the trained model, and present the detected instrument with confidence metrics,
                waveform visualization, and downloadable outputs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <div class="footer-title">InstruNet AI</div>
        <div>
            A professional deep learning project for automatic musical instrument detection using audio signal analysis,
            spectrogram-based preprocessing, and CNN-powered classification.
        </div>
        <div class="footer-sub">
            Built with Streamlit, TensorFlow, Librosa, Matplotlib, SQLite, and ReportLab
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- APP ROUTING ----------------
if st.session_state.logged_in:
    show_main_app()
else:
    if st.session_state.auth_mode == "register":
        show_register_page()
    elif st.session_state.auth_mode == "login":
        show_login_page()
    else:
        show_landing_page()
