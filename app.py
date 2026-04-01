import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import os, json, datetime, tempfile, cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from fpdf import FPDF


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SR         = 22050
DURATION   = 3.0
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
FMIN       = 20
FMAX       = 8000
IMG_H      = 128
IMG_W      = 128
TOP_DB     = 80.0
THRESHOLD  = 0.5

INSTRUMENTS = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi']
INSTRUMENT_NAMES = {
    'cel':'Cello',          'cla':'Clarinet',       'flu':'Flute',
    'gac':'Acoustic Guitar','gel':'Electric Guitar', 'org':'Organ',
    'pia':'Piano',          'sax':'Saxophone',       'tru':'Trumpet',
    'vio':'Violin',         'voi':'Voice'
}
INSTRUMENT_EMOJI = {
    'cel':'🎻','cla':'🎷','flu':'🪈','gac':'🎸','gel':'🎸',
    'org':'🎹','pia':'🎹','sax':'🎷','tru':'🎺','vio':'🎻','voi':'🎤'
}

MODEL_PATH = '/content/drive/MyDrive/IRMAS/models/best_model.h5'

# ─────────────────────────────────────────────────────────────
# USER CREDENTIALS  (add more users here as needed)
# ─────────────────────────────────────────────────────────────

USERS = {
    "aiswarya": "irmas2024",
    "admin": "admin123",
    "demo": "demo123"
}


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎵 IRMAS Instrument Recognizer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  (covers both login page and main app)
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

/* ── LOGIN PAGE ── */
.login-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 80vh;
}
.login-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border: 1px solid #e94560;
    border-radius: 18px;
    padding: 44px 48px 36px;
    max-width: 420px;
    width: 100%;
    box-shadow: 0 8px 40px rgba(233,69,96,0.18);
}
.login-logo {
    text-align: center;
    font-size: 3rem;
    margin-bottom: 4px;
}
.login-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    color: #e94560;
    text-align: center;
    letter-spacing: 4px;
    font-weight: 700;
    margin-bottom: 2px;
}
.login-sub {
    color: #888;
    text-align: center;
    font-size: 0.82rem;
    margin-bottom: 28px;
    letter-spacing: 1px;
}
.login-divider {
    border: none;
    border-top: 1px solid #2a2a2a;
    margin: 0 0 22px 0;
}

/* ── MAIN APP ── */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    border: 1px solid #e94560;
    text-align: center;
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    color: #e94560;
    margin: 0;
    letter-spacing: 2px;
}
.main-header p {
    color: #a0aec0;
    font-size: 1rem;
    margin-top: 8px;
}

/* Cards */
.card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #e94560;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 10px;
}

/* Instrument badges */
.instrument-badge {
    display: inline-block;
    background: linear-gradient(90deg, #e94560, #c0392b);
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    margin: 4px;
}
.instrument-badge-dim {
    display: inline-block;
    background: #2a2a2a;
    color: #666;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    margin: 4px;
}

/* Confidence bars */
.conf-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 10px;
}
.conf-label {
    width: 160px;
    font-size: 0.9rem;
    color: #e0e0e0;
    flex-shrink: 0;
}
.conf-bar-outer {
    flex: 1;
    background: #2a2a2a;
    border-radius: 6px;
    height: 14px;
    overflow: hidden;
}
.conf-bar-inner-high {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #e94560, #ff6b6b);
}
.conf-bar-inner-low {
    height: 100%;
    border-radius: 6px;
    background: #3a3a3a;
}
.conf-val {
    width: 52px;
    text-align: right;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #aaa;
}

/* Step labels */
.step-label {
    background: #e94560;
    color: white;
    border-radius: 50%;
    width: 28px; height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 10px;
    font-family: 'Space Mono', monospace;
}

/* Upload zone */
.stFileUploader > div {
    border: 2px dashed #e94560 !important;
    border-radius: 12px !important;
    background: #111 !important;
}

/* Buttons */
.stDownloadButton > button {
    background: linear-gradient(90deg, #e94560, #c0392b) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 10px 20px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid #2a2a2a;
}

/* Misc */
.stAlert { border-radius: 10px !important; }
.stSpinner { color: #e94560; }

.metric-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #e94560;
    font-weight: 700;
}
.metric-lbl {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Logout button in sidebar */
.logout-btn > button {
    background: transparent !important;
    border: 1px solid #e94560 !important;
    color: #e94560 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
    margin-top: 6px;
}
.logout-btn > button:hover {
    background: #e94560 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""


# ─────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────

def show_login_page():
    _, center_col, _ = st.columns([1, 1.6, 1])
    with center_col:
        st.markdown('<div class="login-logo">🎵</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-title">IRMAS</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">INSTRUMENT RECOGNITION SYSTEM</div>', unsafe_allow_html=True)
        st.markdown('<hr class="login-divider">', unsafe_allow_html=True)

        username = st.text_input("👤 Username", placeholder="Enter your username", key="login_user")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_pass")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("LOGIN  →", use_container_width=True, key="login_btn"):
            if username in USERS and USERS[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"]  = username
                st.success(f"✅ Welcome, {username}!  Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")

        st.markdown("""
        <p style='color:#444; text-align:center; font-size:0.76rem; margin-top:18px;'>
            Demo &nbsp;|&nbsp; username: <b style='color:#888'>demo</b>
            &nbsp;&nbsp; password: <b style='color:#888'>demo123</b>
        </p>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# GATE: show login if not authenticated
# ─────────────────────────────────────────────────────────────

if not st.session_state["logged_in"]:
    show_login_page()
    st.stop()   # ← nothing below runs until user logs in


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def load_and_preprocess(path):
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y, _ = librosa.effects.trim(y, top_db=20)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, SR

def segment_audio(y):
    seg_len = int(DURATION * SR)
    hop     = int(1.5 * SR)
    segs    = []
    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)))
        segs.append(y)
    else:
        for s in range(0, len(y) - seg_len + 1, hop):
            segs.append(y[s:s + seg_len])
    return segs

def to_mel(y):
    mel     = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
                  hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    log_mel = (log_mel + TOP_DB) / TOP_DB
    return cv2.resize(log_mel, (IMG_W, IMG_H)).astype(np.float32)

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def predict(filepath):
    model = load_model()
    if model is None:
        return None
    y, _   = load_and_preprocess(filepath)
    segs   = segment_audio(y)
    specs  = np.array([to_mel(s) for s in segs])[..., np.newaxis]
    probs  = model.predict(specs, verbose=0)
    return np.mean(probs, axis=0)

def build_conf_bars(agg, threshold):
    sorted_idx = np.argsort(agg)[::-1]
    html = ""
    for i in sorted_idx:
        inst  = INSTRUMENTS[i]
        name  = f"{INSTRUMENT_EMOJI[inst]} {INSTRUMENT_NAMES[inst]}"
        score = float(agg[i])
        pct   = score * 100
        bar_class = "conf-bar-inner-high" if score >= threshold else "conf-bar-inner-low"
        html += f"""
        <div class="conf-row">
            <div class="conf-label">{name}</div>
            <div class="conf-bar-outer">
                <div class="{bar_class}" style="width:{pct:.1f}%"></div>
            </div>
            <div class="conf-val">{score:.3f}</div>
        </div>"""
    return html

def export_json(filename, detected, agg):
    result = {
        "file": filename,
        "detected_instruments": detected,
        "confidence_scores": {INSTRUMENTS[i]: round(float(agg[i]), 4) for i in range(11)},
        "threshold_used": THRESHOLD,
        "exported_at": datetime.datetime.now().isoformat()
    }
    return json.dumps(result, indent=2)

def export_pdf(filename, detected, agg):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(13, 13, 13)
    pdf.rect(0, 0, 210, 297, 'F')

    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 14, 'IRMAS Instrument Recognition Report', ln=True, align='C')

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(160, 174, 192)
    pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(4)

    pdf.set_draw_color(233, 69, 96)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 10, 'File Information', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(220, 220, 220)
    pdf.cell(0, 8, f"  Filename  : {filename}", ln=True)
    pdf.cell(0, 8, f"  Threshold : {THRESHOLD}", ln=True)
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 10, 'Detected Instruments', ln=True)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(220, 220, 220)
    if detected:
        for inst in detected:
            pdf.cell(0, 8, f"  [PRESENT]  {INSTRUMENT_NAMES[inst]}", ln=True)
    else:
        pdf.cell(0, 8, "  No instrument detected above threshold.", ln=True)
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 10, 'Confidence Scores', ln=True)
    pdf.set_font('Courier', '', 10)
    pdf.set_text_color(200, 200, 200)
    for inst, score in sorted(
        {INSTRUMENTS[i]: float(agg[i]) for i in range(11)}.items(),
        key=lambda x: -x[1]
    ):
        bar    = '|' * int(score * 30)
        marker = '<<' if score >= THRESHOLD else '  '
        pdf.cell(0, 7, f"  {INSTRUMENT_NAMES[inst]:20s}  {score:.3f}  {bar} {marker}", ln=True)

    import io
    buf = io.BytesIO()
    buf.write(pdf.output(dest='S'))
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────
# SIDEBAR  (shown only when logged in)
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 10px;'>
        <div style='font-size:2.5rem'>🎵</div>
        <div style='font-family:Space Mono; font-size:1rem; color:#e94560; font-weight:700;'>IRMAS</div>
        <div style='font-size:0.78rem; color:#666; margin-top:4px;'>Instrument Recognizer</div>
    </div>
    <hr style='border-color:#2a2a2a; margin:12px 0;'>
    """, unsafe_allow_html=True)

    # ── Logged-in user info ──
    st.markdown(
        f"<div style='background:#1a1a1a; border:1px solid #2a2a2a; border-radius:8px;"
        f"padding:10px 14px; margin-bottom:12px;'>"
        f"<span style='color:#888; font-size:0.78rem;'>LOGGED IN AS</span><br>"
        f"<span style='color:#e94560; font-family:Space Mono; font-weight:700;'>"
        f"👤 {st.session_state['username'].upper()}</span></div>",
        unsafe_allow_html=True
    )

    # ── Logout button ──
    st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
        st.session_state["logged_in"] = False
        st.session_state["username"]  = ""
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2a2a2a; margin:14px 0;'>", unsafe_allow_html=True)

    st.markdown("**⚙️ Settings**")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, THRESHOLD, 0.05,
                          help="Lower = more sensitive. Higher = more strict.")
    THRESHOLD = threshold

    st.markdown("<hr style='border-color:#2a2a2a;'>", unsafe_allow_html=True)
    st.markdown("**🎼 Detectable Instruments**")
    for inst in INSTRUMENTS:
        st.markdown(f"&nbsp;&nbsp;{INSTRUMENT_EMOJI[inst]} {INSTRUMENT_NAMES[inst]}")

    st.markdown("<hr style='border-color:#2a2a2a;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#555; line-height:1.6;'>
    Pipeline:<br>
    Audio → Mono → Resample<br>
    → Trim → Normalize<br>
    → Segment → Mel Spec<br>
    → CNN → Predict
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🎵 INSTRUMENT RECOGNITION SYSTEM</h1>
    <p>Upload an audio file → Detect instruments → Export results</p>
</div>
""", unsafe_allow_html=True)

model_exists = os.path.exists(MODEL_PATH)
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{'✅' if model_exists else '❌'}</div>
        <div class="metric-lbl">Model Status</div>
    </div>""", unsafe_allow_html=True)
with col_s2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">11</div>
        <div class="metric-lbl">Instrument Classes</div>
    </div>""", unsafe_allow_html=True)
with col_s3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{threshold:.2f}</div>
        <div class="metric-lbl">Threshold</div>
    </div>""", unsafe_allow_html=True)

if not model_exists:
    st.error("⚠️ Model not found at: `" + MODEL_PATH + "`  \nTrain the model first (Notebook 4), then re-run the app.")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<span class="step-label">1</span> **Upload Your Audio File**', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag & drop or click to upload",
    type=["wav", "mp3", "flac", "ogg"],
    label_visibility="collapsed"
)

if not uploaded:
    st.markdown("""
    <div class="card" style="text-align:center; padding:40px; color:#555;">
        <div style="font-size:3rem">🎧</div>
        <div style="margin-top:12px; font-size:1rem;">No file uploaded yet</div>
        <div style="font-size:0.85rem; margin-top:6px;">Supported: WAV · MP3 · FLAC · OGG</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

st.markdown('<span class="step-label">2</span> **Audio Preview & Visualization**', unsafe_allow_html=True)
st.audio(uploaded)

y, sr = load_and_preprocess(tmp_path)
duration_sec = len(y) / sr
segs = segment_audio(y)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{duration_sec:.1f}s</div><div class="metric-lbl">Duration</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{SR//1000}kHz</div><div class="metric-lbl">Sample Rate</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{len(segs)}</div><div class="metric-lbl">Segments</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-box"><div class="metric-val">Mono</div><div class="metric-lbl">Channel</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_wave, col_mel = st.columns(2)

with col_wave:
    st.markdown('<div class="card"><div class="card-title">📈 Waveform (Time Domain)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor('#161616')
    ax.set_facecolor('#161616')
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#e94560', alpha=0.85)
    ax.set_xlabel("Time (s)", color='#aaa', fontsize=9)
    ax.set_ylabel("Amplitude", color='#aaa', fontsize=9)
    ax.tick_params(colors='#555')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with col_mel:
    st.markdown('<div class="card"><div class="card-title">🌈 Mel Spectrogram (Frequency Domain)</div>', unsafe_allow_html=True)
    mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor('#161616')
    ax.set_facecolor('#161616')
    img = librosa.display.specshow(log_mel, sr=sr, hop_length=HOP_LENGTH,
             x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#aaa')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaa', fontsize=8)
    ax.set_xlabel("Time (s)", color='#aaa', fontsize=9)
    ax.set_ylabel("Mel Freq", color='#aaa', fontsize=9)
    ax.tick_params(colors='#555')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<span class="step-label">3</span> **Run Instrument Detection**', unsafe_allow_html=True)

if st.button("🎯 Detect Instruments", use_container_width=True):
    with st.spinner("Analyzing audio through CNN model..."):
        agg = predict(tmp_path)

    if agg is None:
        st.error("Model could not be loaded. Check model path.")
        st.stop()

    detected = [INSTRUMENTS[i] for i in range(11) if agg[i] >= THRESHOLD]

    st.markdown("<br>", unsafe_allow_html=True)

    if detected:
        names_str = "  ·  ".join([f"{INSTRUMENT_EMOJI[d]} {INSTRUMENT_NAMES[d]}" for d in detected])
        st.success(f"**Detected:** {names_str}")
    else:
        st.warning("No instrument confidently detected above the threshold. Try lowering the threshold in the sidebar.")

    st.markdown('<div class="card"><div class="card-title">🎼 Detection Result</div>', unsafe_allow_html=True)
    badge_html = ""
    for inst in INSTRUMENTS:
        if agg[INSTRUMENTS.index(inst)] >= THRESHOLD:
            badge_html += f'<span class="instrument-badge">{INSTRUMENT_EMOJI[inst]} {INSTRUMENT_NAMES[inst]}</span>'
        else:
            badge_html += f'<span class="instrument-badge-dim">{INSTRUMENT_NAMES[inst]}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📊 Confidence Scores</div>', unsafe_allow_html=True)
    st.markdown(build_conf_bars(agg, THRESHOLD), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📋 Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    top_inst  = INSTRUMENTS[int(np.argmax(agg))]
    top_score = float(np.max(agg))
    c1.metric("Detected",       str(len(detected)))
    c2.metric("Top Instrument", INSTRUMENT_NAMES[top_inst])
    c3.metric("Top Score",      f"{top_score:.3f}")
    c4.metric("Threshold",      f"{THRESHOLD:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<span class="step-label">4</span> **Export Results**', unsafe_allow_html=True)
    exp1, exp2 = st.columns(2)

    with exp1:
        json_str = export_json(uploaded.name, detected, agg)
        st.download_button(
            label="⬇️ Download JSON Report",
            data=json_str,
            file_name=f"irmas_{uploaded.name.split('.')[0]}.json",
            mime="application/json",
            use_container_width=True
        )

    with exp2:
        pdf_bytes = export_pdf(uploaded.name, detected, agg)
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"irmas_{uploaded.name.split('.')[0]}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with st.expander("👁️ Preview JSON Output"):
        st.code(json_str, language="json")

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:30px; color:#555;">
        <div style="font-size:2rem">🎯</div>
        <div style="margin-top:8px;">Click the button above to run detection</div>
    </div>
    """, unsafe_allow_html=True)

try:
    os.unlink(tmp_path)
except:
    pass
