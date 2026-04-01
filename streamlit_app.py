# ============================================
# 🎵 InstruNet AI - Streamlit App
# CNN-Based Music Instrument Recognition System
# Converted from Flask → Streamlit
# ============================================

import os
import io
import json
import numpy as np
import librosa
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="InstruNet AI — Music Instrument Recognition",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================
# CUSTOM CSS — Premium Dark Glassmorphism
# ============================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ---- Base ---- */
    .stApp {
        background: #0a0e1a !important;
        color: #e6edf3 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ---- Animated gradient blurs ---- */
    .bg-blur-1, .bg-blur-2 {
        position: fixed; border-radius: 50%; filter: blur(120px); opacity: 0.25;
        pointer-events: none; z-index: 0;
    }
    .bg-blur-1 { width: 600px; height: 600px; top: -120px; left: -80px;
        background: radial-gradient(circle, #6366f1 0%, transparent 70%);
        animation: float1 18s ease-in-out infinite; }
    .bg-blur-2 { width: 500px; height: 500px; bottom: -100px; right: -60px;
        background: radial-gradient(circle, #f97316 0%, transparent 70%);
        animation: float2 22s ease-in-out infinite; }

    @keyframes float1 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(60px,40px)} }
    @keyframes float2 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-50px,-30px)} }

    /* ---- Glass Card ---- */
    .glass-card {
        background: rgba(22, 27, 45, 0.65);
        backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 20px; padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative; z-index: 1;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }

    /* ---- Hero ---- */
    .hero-title {
        font-size: 3.2rem; font-weight: 900;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 30%, #f97316 70%, #fb923c 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.3rem;
        letter-spacing: -0.03em; line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 1.15rem; color: #94a3b8; font-weight: 400; margin-bottom: 1.5rem;
    }
    .hero-desc {
        font-size: 0.92rem; color: #64748b; line-height: 1.6; max-width: 650px;
    }

    /* ---- Section Headers ---- */
    .section-header {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
        text-transform: uppercase; color: #6366f1; margin-top: 2.5rem;
        margin-bottom: 1rem; padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* ---- Metric Cards ---- */
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 45, 0.8) 0%, rgba(30, 36, 58, 0.8) 100%);
        backdrop-filter: blur(12px); border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px; padding: 1.3rem 1.6rem;
        position: relative; overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #6366f1, #f97316); border-radius: 16px 16px 0 0;
    }
    .metric-card:hover { border-color: #6366f1; transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(99, 102, 241, 0.18); }
    .metric-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #8b949e; margin-bottom: 0.25rem; }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #e6edf3; line-height: 1.15; }
    .metric-sub { font-size: 0.72rem; color: #818cf8; margin-top: 0.2rem; font-weight: 500; }

    /* ---- Instrument Result Cards ---- */
    .instrument-card {
        background: rgba(22, 27, 45, 0.7);
        backdrop-filter: blur(10px); border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 14px; padding: 1rem 1.3rem;
        transition: all 0.3s ease;
    }
    .instrument-card:hover { transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.15); }
    .instrument-card.present { border-left: 3px solid #22c55e; }
    .instrument-card.absent  { border-left: 3px solid #ef4444; opacity: 0.7; }

    .instrument-name { font-size: 1rem; font-weight: 700; color: #e6edf3; margin-bottom: 0.3rem; }
    .badge-green {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        background: rgba(34, 197, 94, 0.15); color: #22c55e; letter-spacing: 0.05em;
    }
    .badge-red {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        background: rgba(239, 68, 68, 0.15); color: #ef4444; letter-spacing: 0.05em;
    }
    .confidence-text { font-size: 0.82rem; color: #94a3b8; margin: 0.3rem 0 0.5rem; }
    .progress-bar-bg {
        width: 100%; height: 6px; background: rgba(99, 102, 241, 0.1);
        border-radius: 3px; overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%; border-radius: 3px;
        background: linear-gradient(90deg, #6366f1, #818cf8);
        transition: width 0.8s ease;
    }
    .progress-bar-fill.red {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }

    /* ---- Top Prediction ---- */
    .top-prediction-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(249, 115, 22, 0.12) 100%);
        backdrop-filter: blur(16px); border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px; padding: 2.5rem; text-align: center;
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }
    .top-pred-label { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.15em;
        text-transform: uppercase; color: #f97316; margin-bottom: 0.5rem; }
    .top-pred-name {
        font-size: 2.8rem; font-weight: 900;
        background: linear-gradient(135deg, #818cf8, #f97316);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1.15; margin-bottom: 0.4rem;
    }
    .top-pred-conf { font-size: 1.2rem; font-weight: 600; color: #a5b4fc; }

    /* ---- Audio Player ---- */
    .audio-section { margin: 1rem 0; }
    audio { width: 100%; border-radius: 10px; }

    /* ---- Navigation Tabs ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: #161b2e; border-radius: 10px; color: #8b949e;
        border: 1px solid rgba(99, 102, 241, 0.15); font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important; border-color: #6366f1 !important;
    }

    /* ---- File Uploader ---- */
    section[data-testid="stFileUploader"] {
        background: rgba(22, 27, 45, 0.6); border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 16px; padding: 1.2rem;
        transition: border-color 0.3s ease;
    }
    section[data-testid="stFileUploader"]:hover { border-color: #6366f1; }

    /* ---- Hide defaults ---- */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #0d1220 !important; border-right: 1px solid rgba(99,102,241,0.12) !important;
    }

    /* ---- Footer ---- */
    .app-footer {
        text-align: center; color: #475569; font-size: 0.72rem; padding: 2rem 0 1rem;
        border-top: 1px solid rgba(99,102,241,0.1); margin-top: 3rem;
    }
</style>
"""

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
    "cel": "Cello",    "cla": "Clarinet",  "flu": "Flute",
    "gac": "Acoustic Guitar", "gel": "Electric Guitar",
    "org": "Organ",    "pia": "Piano",     "sax": "Saxophone",
    "tru": "Trumpet",  "vio": "Violin",    "voi": "Voice",
}

INSTRUMENT_COLORS = [
    "#818cf8", "#22c55e", "#3b82f6", "#d8b4fe",
    "#f97316", "#67e8f9", "#86efac", "#fb923c",
    "#f87171", "#7dd3fc", "#f472b6",
]

INSTRUMENT_EMOJIS = {
    "Cello": "🎻", "Clarinet": "🎶", "Flute": "🪈",
    "Acoustic Guitar": "🎸", "Electric Guitar": "🎸",
    "Organ": "🎹", "Piano": "🎹", "Saxophone": "🎷",
    "Trumpet": "🎺", "Violin": "🎻", "Voice": "🎤",
}

# ============================================
# LOAD MODEL (Cached)
# ============================================
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models", "instrunet_cnn.h5")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "models", "label_classes.json")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH, "r") as f:
            classes = json.load(f)
        return model, classes, True
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, list(INSTRUMENT_NAMES.keys()), False

# ============================================
# AUDIO PROCESSING HELPERS
# ============================================
def load_audio(audio_bytes, sr=SAMPLE_RATE):
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
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

def predict_instruments(model, audio, sr=SAMPLE_RATE):
    segment_len = sr * SEGMENT_SECS
    segments = [
        audio[i:i + segment_len]
        for i in range(0, len(audio), segment_len)
        if len(audio[i:i + segment_len]) >= sr
    ]
    if not segments:
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
# PLOTLY CHART HELPERS
# ============================================
def plotly_theme(fig, height=400, title="", x_title="", y_title="", show_legend=True):
    fig.update_layout(
        paper_bgcolor="#0d1220", plot_bgcolor="#0a0e1a",
        font=dict(family="Inter", color="#e6edf3", size=12),
        margin=dict(l=55, r=20, t=55, b=50),
        hoverlabel=dict(bgcolor="#161b2e", font_color="#e6edf3", bordercolor="#30363d"),
        height=height, showlegend=show_legend,
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#e6edf3")))
    fig.update_xaxes(gridcolor="#1e2438", zerolinecolor="#1e2438", title=x_title if x_title else None)
    fig.update_yaxes(gridcolor="#1e2438", zerolinecolor="#1e2438", title=y_title if y_title else None)
    return fig

def create_waveform_chart(audio, sr):
    t = np.linspace(0, len(audio) / sr, len(audio))
    # Downsample for performance
    step = max(1, len(audio) // 8000)
    t_ds, audio_ds = t[::step], audio[::step]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_ds, y=audio_ds, mode="lines",
        line=dict(width=1, color="#818cf8"),
        fill="tozeroy", fillcolor="rgba(129, 140, 248, 0.08)",
        hovertemplate="<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.4f}<extra></extra>",
    ))
    fig = plotly_theme(fig, height=300, title="📈 Waveform", x_title="Time (s)", y_title="Amplitude", show_legend=False)
    return fig

def create_spectrogram_chart(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    times = librosa.times_like(mel_db, sr=sr, hop_length=512)
    freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)

    fig = go.Figure(go.Heatmap(
        z=mel_db, x=times, y=freqs,
        colorscale="Magma", colorbar=dict(title=dict(text="dB", font=dict(color="#8b949e")), tickfont=dict(color="#8b949e")),
        hovertemplate="<b>Time:</b> %{x:.2f}s<br><b>Freq:</b> %{y:.0f} Hz<br><b>Power:</b> %{z:.1f} dB<extra></extra>",
    ))
    fig = plotly_theme(fig, height=350, title="🌈 Mel-Spectrogram", x_title="Time (s)", y_title="Frequency (Hz)", show_legend=False)
    return fig

def create_timeline_chart(timeline, classes):
    readable = [INSTRUMENT_NAMES.get(c, c) for c in classes]
    n_seg = timeline.shape[0]
    x = np.arange(n_seg) * SEGMENT_SECS

    fig = go.Figure()
    for i, (name, color) in enumerate(zip(readable, INSTRUMENT_COLORS[:len(readable)])):
        fig.add_trace(go.Scatter(
            x=x, y=timeline[:, i] * 100, mode="lines+markers",
            name=name, line=dict(width=2.5, color=color),
            marker=dict(size=5),
            hovertemplate=f"<b>{name}</b><br>Time: %{{x}}s<br>Confidence: %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_hline(y=THRESHOLD * 100, line=dict(color="#fbbf24", width=2, dash="dash"),
                  annotation_text=f"Threshold ({THRESHOLD*100:.0f}%)",
                  annotation_font=dict(color="#fbbf24", size=11))
    fig = plotly_theme(fig, height=420, title="⏱ Instrument Activity Over Time",
                       x_title="Time (s)", y_title="Confidence (%)")
    fig.update_layout(legend=dict(font=dict(size=10, color="#e6edf3"), bgcolor="rgba(0,0,0,0)"))
    return fig

def create_prediction_bar(predictions, threshold):
    pred_sorted = sorted(predictions, key=lambda p: p["confidence"])
    fig = go.Figure(go.Bar(
        y=[p["name"] for p in pred_sorted],
        x=[p["confidence"] for p in pred_sorted],
        orientation="h",
        marker=dict(
            color=[
                "#22c55e" if p["confidence"] >= threshold * 100 else "#ef4444"
                for p in pred_sorted
            ],
            line=dict(width=0),
        ),
        text=[f'{p["confidence"]}%' for p in pred_sorted],
        textposition="outside",
        textfont=dict(size=11, color="#e6edf3"),
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=threshold * 100, line=dict(color="#fbbf24", width=2, dash="dash"),
                  annotation_text=f"Threshold ({threshold*100:.0f}%)",
                  annotation_font=dict(color="#fbbf24", size=10))
    fig = plotly_theme(fig, height=450, title="🎯 All Instrument Predictions",
                       x_title="Confidence (%)", show_legend=False)
    fig.update_xaxes(range=[0, 105])
    fig.update_yaxes(title="")
    return fig

# ============================================
# HTML COMPONENT HELPERS
# ============================================
def render_metric_card(label, value, sub):
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""

def render_instrument_card(name, confidence, present, threshold_pct):
    cls = "present" if present else "absent"
    badge = '<span class="badge-green">Present</span>' if present else '<span class="badge-red">Not Present</span>'
    emoji = INSTRUMENT_EMOJIS.get(name, "🎵")
    bar_class = "" if present else "red"
    return f"""<div class="instrument-card {cls}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="instrument-name">{emoji} {name}</span>
            {badge}
        </div>
        <div class="confidence-text">Confidence: <strong>{confidence}%</strong></div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill {bar_class}" style="width:{confidence}%;"></div>
        </div>
    </div>"""

# ============================================
# MAIN APP
# ============================================
def main():
    # Inject CSS + animated blurs
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="bg-blur-1"></div><div class="bg-blur-2"></div>', unsafe_allow_html=True)

    # Load model
    model, classes, model_loaded = load_model()

    # ========================
    # HERO HEADER
    # ========================
    st.markdown("""
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <div class="hero-title">🎵 InstruNet AI</div>
            <div class="hero-subtitle">CNN-Based Music Instrument Recognition System</div>
            <div class="hero-desc" style="margin:0 auto;">
                Upload an audio file and let the AI identify musical instruments
                using waveform analysis, mel-spectrograms, and deep learning.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ========================
    # TABS (Replace Slide Navigation)
    # ========================
    if "result" not in st.session_state:
        st.session_state.result = None

    # Upload Section
    st.markdown('<div class="section-header">📂 Upload Audio</div>', unsafe_allow_html=True)

    upload_col1, upload_col2 = st.columns([1.5, 1])

    with upload_col1:
        uploaded_file = st.file_uploader(
            "Upload an audio file for analysis",
            type=["wav", "mp3", "flac"],
            help="Supported formats: WAV, MP3, FLAC (max 25MB)",
        )

    with upload_col2:
        st.markdown("""
            <div class="glass-card" style="padding:1.3rem; margin-top:0.5rem;">
                <div style="font-size:0.75rem; font-weight:700; color:#6366f1; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;">Supported Formats</div>
                <div style="color:#94a3b8; font-size:0.85rem; line-height:1.8;">
                    🎵 <strong style="color:#e6edf3;">WAV</strong> — Uncompressed audio<br>
                    🎶 <strong style="color:#e6edf3;">MP3</strong> — Compressed audio<br>
                    🎼 <strong style="color:#e6edf3;">FLAC</strong> — Lossless audio<br>
                    📏 <strong style="color:#e6edf3;">Max size:</strong> 25 MB
                </div>
            </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        # Audio preview
        st.markdown("")
        st.audio(uploaded_file, format="audio/wav")
        st.markdown("")

        # Analyze button
        analyze = st.button("🚀 Analyze Audio", type="primary", use_container_width=True)

        if analyze:
            with st.spinner("🔍 Analyzing audio... This may take a moment."):
                audio_bytes = uploaded_file.read()
                uploaded_file.seek(0)

                audio = load_audio(audio_bytes)
                duration = len(audio) / SAMPLE_RATE

                # Run prediction
                if model is not None:
                    avg_pred, timeline = predict_instruments(model, audio)
                    predictions = []
                    for i, c in enumerate(classes):
                        confidence = round(float(avg_pred[i]) * 100, 2)
                        present = bool(avg_pred[i] >= THRESHOLD)
                        predictions.append({
                            "code": c,
                            "name": INSTRUMENT_NAMES.get(c, c),
                            "confidence": confidence,
                            "present": present,
                        })
                    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
                else:
                    timeline = None
                    predictions = []

                st.session_state.result = {
                    "audio": audio,
                    "duration": round(duration, 2),
                    "sample_rate": SAMPLE_RATE,
                    "predictions": predictions,
                    "timeline": timeline,
                    "classes": classes,
                    "filename": uploaded_file.name,
                }

    # ========================
    # RESULTS
    # ========================
    result = st.session_state.result

    if result is not None:
        audio = result["audio"]
        predictions = result["predictions"]
        timeline = result["timeline"]

        st.markdown("---")

        # Tabs for organized content
        tab1, tab2, tab3 = st.tabs(["📊 Audio Analysis", "🏆 Predictions", "⏱ Timeline"])

        # ========== TAB 1: Audio Analysis ==========
        with tab1:
            # Audio info cards
            st.markdown('<div class="section-header">🎧 Audio Information</div>', unsafe_allow_html=True)

            info_cols = st.columns(3)
            with info_cols[0]:
                st.markdown(render_metric_card("📁 File", result["filename"], "Uploaded audio"), unsafe_allow_html=True)
            with info_cols[1]:
                st.markdown(render_metric_card("⏱ Duration", f'{result["duration"]}s', f'{result["sample_rate"]} Hz sample rate'), unsafe_allow_html=True)
            with info_cols[2]:
                n_segments = timeline.shape[0] if timeline is not None else 0
                st.markdown(render_metric_card("🔢 Segments", str(n_segments), f"{SEGMENT_SECS}s per segment"), unsafe_allow_html=True)

            st.markdown("")

            # Waveform + Spectrogram
            st.markdown('<div class="section-header">📊 Audio Visualizations</div>', unsafe_allow_html=True)

            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                waveform_fig = create_waveform_chart(audio, SAMPLE_RATE)
                st.plotly_chart(waveform_fig, use_container_width=True)

            with viz_col2:
                spec_fig = create_spectrogram_chart(audio, SAMPLE_RATE)
                st.plotly_chart(spec_fig, use_container_width=True)

        # ========== TAB 2: Predictions ==========
        with tab2:
            if predictions:
                # Top Prediction
                top = predictions[0]
                emoji = INSTRUMENT_EMOJIS.get(top["name"], "🎵")
                st.markdown(f"""
                    <div class="top-prediction-card">
                        <div class="top-pred-label">🏆 Top Prediction</div>
                        <div class="top-pred-name">{emoji} {top["name"]}</div>
                        <div class="top-pred-conf">{top["confidence"]}% Confidence</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                # Top 3
                st.markdown('<div class="section-header">🥇 Top 3 Predictions</div>', unsafe_allow_html=True)
                top3_cols = st.columns(3)
                medals = ["🥇", "🥈", "🥉"]
                for idx, col in enumerate(top3_cols):
                    if idx < len(predictions):
                        p = predictions[idx]
                        with col:
                            st.markdown(f"""
                                <div class="glass-card" style="text-align:center; padding:1.5rem;">
                                    <div style="font-size:2rem; margin-bottom:0.3rem;">{medals[idx]}</div>
                                    <div style="font-size:1.15rem; font-weight:800; color:#e6edf3;">{INSTRUMENT_EMOJIS.get(p["name"], "🎵")} {p["name"]}</div>
                                    <div style="font-size:1.5rem; font-weight:900; color:#818cf8; margin:0.4rem 0;">{p["confidence"]}%</div>
                                    <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{p['confidence']}%;"></div></div>
                                </div>
                            """, unsafe_allow_html=True)

                st.markdown("")

                # Full prediction bar chart
                st.markdown('<div class="section-header">🎯 Full Prediction Breakdown</div>', unsafe_allow_html=True)
                bar_fig = create_prediction_bar(predictions, THRESHOLD)
                st.plotly_chart(bar_fig, use_container_width=True)

                st.markdown("")

                # Present vs Absent
                present = [p for p in predictions if p["present"]]
                absent  = [p for p in predictions if not p["present"]]

                pres_col, abs_col = st.columns(2)

                with pres_col:
                    st.markdown('<div class="section-header">✅ Present Instruments</div>', unsafe_allow_html=True)
                    if present:
                        for p in present:
                            st.markdown(render_instrument_card(p["name"], p["confidence"], True, THRESHOLD * 100), unsafe_allow_html=True)
                            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                    else:
                        st.info("No instruments detected above the threshold.")

                with abs_col:
                    st.markdown('<div class="section-header">❌ Not Present</div>', unsafe_allow_html=True)
                    if absent:
                        for p in absent:
                            st.markdown(render_instrument_card(p["name"], p["confidence"], False, THRESHOLD * 100), unsafe_allow_html=True)
                            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                    else:
                        st.info("All instruments detected above the threshold!")
            else:
                st.warning("⚠️ No predictions available. Model may not be loaded.")

        # ========== TAB 3: Timeline ==========
        with tab3:
            if timeline is not None:
                st.markdown('<div class="section-header">⏱ Instrument Activity Timeline</div>', unsafe_allow_html=True)
                st.markdown("""
                    <div style="color:#64748b; font-size:0.85rem; margin-bottom:1rem;">
                        Confidence of each detected instrument across time segments of the audio.
                        The dashed line indicates the detection threshold.
                    </div>
                """, unsafe_allow_html=True)
                timeline_fig = create_timeline_chart(timeline, result["classes"])
                st.plotly_chart(timeline_fig, use_container_width=True)

                # Per-segment breakdown
                st.markdown("")
                st.markdown('<div class="section-header">📋 Segment Details</div>', unsafe_allow_html=True)

                readable = [INSTRUMENT_NAMES.get(c, c) for c in result["classes"]]
                seg_data = {}
                for seg_i in range(timeline.shape[0]):
                    seg_data[f"Seg {seg_i+1} ({seg_i * SEGMENT_SECS}-{(seg_i+1) * SEGMENT_SECS}s)"] = {
                        name: f"{timeline[seg_i, j]*100:.1f}%" for j, name in enumerate(readable)
                    }

                import pandas as pd
                seg_df = pd.DataFrame(seg_data).T
                st.dataframe(seg_df, use_container_width=True)
            else:
                st.warning("⚠️ Timeline data not available.")

    # Footer
    st.markdown("""
        <div class="app-footer">
            🎵 InstruNet AI • CNN-Based Music Instrument Recognition • TensorFlow/Keras • Streamlit
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()