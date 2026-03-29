import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import tempfile
import json
import os
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e2e2f0;
}

.stApp { background: #07070f; }

[data-testid="stSidebar"] {
    background: #0c0c1a !important;
    border-right: 1px solid #1a1a2e !important;
}
[data-testid="stSidebar"] * { color: #c4c4d8 !important; }

.app-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 1.8rem 0 1.2rem 0;
    border-bottom: 1px solid #1a1a2e;
    margin-bottom: 1.8rem;
}
.app-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(120deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
    line-height: 1;
}
.app-sub {
    font-size: 0.78rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 300;
    margin-top: 6px;
}
.app-version {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #374151;
    padding: 4px 10px;
    border: 1px solid #1a1a2e;
    border-radius: 99px;
}

.pipeline {
    display: flex;
    align-items: center;
    margin-bottom: 1.8rem;
    background: #0c0c1a;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    overflow-x: auto;
}
.pipe-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 14px;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #374151;
    white-space: nowrap;
}
.pipe-step.active { background: #1a1230; color: #a78bfa; }
.pipe-step.done   { color: #34d399; }
.pipe-arrow { color: #1f2937; font-size: 0.9rem; padding: 0 4px; }

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #4b5563;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1a2e;
}

.card {
    background: #0e0e1c;
    border: 1px solid #1a1a2e;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}

.empty-box {
    background: #090912;
    border: 1px dashed #1f1f30;
    border-radius: 12px;
    padding: 2.5rem 1rem;
    text-align: center;
    color: #374151;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
}

.inst-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.85rem;
    gap: 10px;
}
.inst-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #9ca3af;
    width: 72px;
    flex-shrink: 0;
    text-transform: capitalize;
}
.bar-track {
    flex: 1;
    height: 6px;
    background: #141425;
    border-radius: 99px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 99px; }
.inst-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #6b7280;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}
.inst-badge {
    font-size: 0.6rem;
    font-family: 'Space Mono', monospace;
    padding: 2px 7px;
    border-radius: 99px;
    background: #0d2218;
    color: #34d399;
    border: 1px solid #34d39930;
    flex-shrink: 0;
    width: 60px;
    text-align: center;
}
.inst-badge-no {
    font-size: 0.6rem;
    font-family: 'Space Mono', monospace;
    padding: 2px 7px;
    border-radius: 99px;
    background: #111120;
    color: #374151;
    border: 1px solid #1f2937;
    flex-shrink: 0;
    width: 60px;
    text-align: center;
}

.stat-box {
    background: #090912;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    padding: 1rem 0.8rem;
    text-align: center;
}
.stat-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #a78bfa;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.65rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 5px;
}

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.6rem 0 1rem 0; }
.chip {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 5px 14px;
    border-radius: 99px;
    border: 1px solid;
}

.tl-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid #111120;
    font-size: 0.78rem;
}
.tl-time {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #4b5563;
    width: 100px;
    flex-shrink: 0;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px #7c3aed44 !important;
}
.stDownloadButton > button {
    background: #0e0e1c !important;
    color: #a78bfa !important;
    border: 1px solid #a78bfa30 !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #16162a !important;
    border-color: #a78bfa80 !important;
}

hr { border-color: #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ───────────────────────────────────────────────────────────────
LABELS = ["keyboard", "guitar", "bass", "string", "brass", "reed", "mallet", "organ", "flute"]

COLORS = {
    "keyboard": "#a78bfa", "guitar": "#60a5fa", "bass":   "#34d399",
    "string":   "#fb923c", "brass":  "#fbbf24", "reed":   "#f472b6",
    "mallet":   "#38bdf8", "organ":  "#c084fc", "flute":  "#86efac",
}
CHIP_COLORS = {
    "keyboard": ("#a78bfa30","#a78bfa"), "guitar": ("#60a5fa30","#60a5fa"),
    "bass":     ("#34d39930","#34d399"), "string": ("#fb923c30","#fb923c"),
    "brass":    ("#fbbf2430","#fbbf24"), "reed":   ("#f472b630","#f472b6"),
    "mallet":   ("#38bdf830","#38bdf8"), "organ":  ("#c084fc30","#c084fc"),
    "flute":    ("#86efac30","#86efac"),
}
INST_DESC = {
    "keyboard": "Keyboard/piano - harmonic sustain, wide pitch range.",
    "guitar":   "Guitar - plucked strings, bright attack.",
    "bass":     "Bass - low-frequency foundation, rhythmic.",
    "string":   "String (violin/cello) - bowed, rich harmonic texture.",
    "brass":    "Brass (trumpet/trombone) - bright, buzzy timbre.",
    "reed":     "Reed (clarinet/sax) - warm woodwind character.",
    "mallet":   "Mallet (xylophone/marimba) - percussive, tonal.",
    "organ":    "Organ - sustained, full-spectrum tone.",
    "flute":    "Flute - airy, pure tone, high-frequency.",
}


# ─── MODEL ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    m    = tf.keras.models.load_model("model.keras")
    mean = np.load("mean.npy")
    std  = np.load("std.npy")
    return m, mean, std

try:
    model, MEAN, STD = load_model_cached()
    model_ok = True
except Exception as e:
    model_ok  = False
    model_err = str(e)


# ─── AUDIO HELPERS ───────────────────────────────────────────────────────────
def load_audio(file_path):
    y, sr    = librosa.load(file_path, sr=16000)
    duration = len(y) / sr
    y, _     = librosa.effects.trim(y)
    rms      = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y * (0.1 / rms)
    return y, sr, duration


def predict_segments(y, sr, threshold):
    SEG_LEN = sr * 3
    FIXED   = 128
    preds, timeline = [], []

    for start in range(0, len(y) - SEG_LEN + 1, SEG_LEN):
        seg    = y[start:start + SEG_LEN]
        mel    = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128, n_fft=2048, hop_length=256)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
        if mel_db.shape[1] < FIXED:
            mel_db = np.pad(mel_db, ((0,0),(0, FIXED - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :FIXED]
        X = (np.expand_dims(np.expand_dims(mel_db, -1), 0) - MEAN) / STD
        pred = model.predict(X, verbose=0)[0]
        preds.append(pred)
        t0 = round(start / sr, 2)
        t1 = round((start + SEG_LEN) / sr, 2)
        timeline.append({
            "start": t0, "end": t1,
            "instruments": [LABELS[i] for i in range(len(LABELS)) if pred[i] >= threshold],
            "confidences": {LABELS[i]: round(float(pred[i]), 4) for i in range(len(LABELS))}
        })

    if not preds:
        return np.zeros(len(LABELS)), [], 0
    return np.mean(preds, axis=0), timeline, len(preds)


def make_vis(y, sr):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), facecolor='#07070f')
    times = np.linspace(0, len(y)/sr, len(y))
    ax1.plot(times, y, color='#a78bfa', lw=0.5, alpha=0.85)
    ax1.set_facecolor('#07070f'); ax1.set_xlim(0, times[-1])
    ax1.tick_params(colors='#374151', labelsize=6)
    ax1.spines[:].set_color('#1a1a2e')
    ax1.set_ylabel('Amplitude', color='#4b5563', fontsize=7)

    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = librosa.display.specshow(mel_db, sr=sr, hop_length=256,
                                       x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
    ax2.set_facecolor('#07070f')
    ax2.tick_params(colors='#374151', labelsize=6)
    ax2.spines[:].set_color('#1a1a2e')
    ax2.set_xlabel('Time (s)', color='#4b5563', fontsize=7)
    ax2.set_ylabel('Hz', color='#4b5563', fontsize=7)
    cb = plt.colorbar(img, ax=ax2, format='%+2.0f dB')
    cb.ax.yaxis.set_tick_params(color='#4b5563', labelsize=6)
    cb.outline.set_edgecolor('#1a1a2e')
    plt.tight_layout(pad=0.8)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor='#07070f', bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return buf


def make_chart(prediction):
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor='white')
    colors  = [COLORS[l] for l in LABELS]
    vals    = [float(prediction[i]) * 100 for i in range(len(LABELS))]
    bars    = ax.barh(LABELS, vals, color=colors, height=0.55)
    ax.set_facecolor('white'); ax.set_xlim(0, 105)
    ax.set_xlabel('Confidence (%)', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', ha='left', fontsize=7)
    plt.tight_layout(pad=0.5)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor='white', bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return buf


# ─── JSON ────────────────────────────────────────────────────────────────────
def build_json(fname, duration, prediction, timeline, threshold, num_segs):
    instruments = {}
    for i, label in enumerate(LABELS):
        p = float(prediction[i])
        instruments[label] = {
            "confidence":     round(p, 4),
            "confidence_pct": f"{p*100:.1f}%",
            "detected":       bool(p >= threshold),
            "description":    INST_DESC[label]
        }
    detected = [l for l in LABELS if instruments[l]["detected"]]
    return json.dumps({
        "metadata": {
            "audio_file":      fname,
            "duration_sec":    round(duration, 2),
            "analyzed_at":     datetime.datetime.now().isoformat(timespec='seconds'),
            "total_segments":  num_segs
        },
        "model_config": {
            "model":                  "InstruNet AI - CNN Mel-Spectrogram",
            "sample_rate":            16000,
            "segment_duration_sec":   3,
            "n_mels":                 128,
            "n_fft":                  2048,
            "hop_length":             256,
            "threshold":              threshold,
            "aggregation":            "mean across segments"
        },
        "summary": {
            "detected_count":        len(detected),
            "detected_instruments":  detected,
            "top_instrument":        LABELS[int(np.argmax(prediction))],
            "top_confidence":        f"{float(np.max(prediction))*100:.1f}%"
        },
        "instrument_predictions": instruments,
        "timeline": timeline
    }, indent=4).encode("utf-8")


# ─── PDF ─────────────────────────────────────────────────────────────────────
def build_pdf(fname, duration, prediction, timeline, threshold, num_segs, spec_buf, chart_buf):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Cover band
    pdf.set_fill_color(10, 10, 20)
    pdf.rect(0, 0, 210, 36, 'F')
    pdf.set_y(7)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(167, 139, 250)
    pdf.cell(0, 10, "InstruNet AI", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 130)
    pdf.cell(0, 6, "CNN-Based Music Instrument Recognition - Analysis Report",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    def sec(title):
        pdf.set_fill_color(20, 18, 40)
        pdf.set_text_color(167, 139, 250)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def kv(k, v):
        pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(80, 80, 110)
        pdf.cell(60, 7, k)
        pdf.set_font("Helvetica", "", 9); pdf.set_text_color(180, 180, 210)
        pdf.cell(0, 7, str(v), new_x="LMARGIN", new_y="NEXT")

    # ── Audio details ──
    sec("AUDIO FILE DETAILS")
    kv("File Name:",      fname)
    kv("Duration:",       f"{round(duration,2)} seconds")
    kv("Analyzed At:",    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    kv("Total Segments:", str(num_segs))
    pdf.ln(4)

    # ── Model config ──
    sec("MODEL CONFIGURATION")
    kv("Model:",          "InstruNet AI - CNN Mel-Spectrogram")
    kv("Sample Rate:",    "16,000 Hz")
    kv("Segment Length:", "3 seconds")
    kv("Mel Bands:",      "128")
    kv("FFT Size:",       "2048")
    kv("Hop Length:",     "256")
    kv("Threshold:",      str(threshold))
    kv("Aggregation:",    "Mean confidence across all segments")
    pdf.ln(4)

    # ── Summary ──
    sec("DETECTED INSTRUMENTS SUMMARY")
    detected = [LABELS[i] for i in range(len(LABELS)) if prediction[i] >= threshold]
    if detected:
        pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(52, 211, 153)
        pdf.cell(0, 8, "  Detected:  " + ",   ".join([d.capitalize() for d in detected]),
                 new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 9); pdf.set_text_color(100, 100, 130)
        pdf.cell(0, 8, "  No instruments detected above threshold.", new_x="LMARGIN", new_y="NEXT")

    top_inst = LABELS[int(np.argmax(prediction))]
    top_conf = float(np.max(prediction)) * 100
    pdf.set_font("Helvetica", "", 8); pdf.set_text_color(100, 100, 130)
    pdf.cell(0, 6, f"  Top instrument: {top_inst.capitalize()}  -  Confidence: {top_conf:.1f}%",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Confidence table ──
    sec("INSTRUMENT CONFIDENCE SCORES")
    pdf.set_fill_color(18, 16, 36); pdf.set_text_color(120, 110, 180)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(55, 7, "  Instrument", border=1, fill=True)
    pdf.cell(38, 7, "Confidence",  border=1, fill=True)
    pdf.cell(32, 7, "Status",      border=1, fill=True)
    pdf.cell(0,  7, "Description", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for i, label in enumerate(LABELS):
        p  = float(prediction[i])
        ok = p >= threshold
        if ok:
            pdf.set_fill_color(12, 28, 20)
        else:
            pdf.set_fill_color(12, 12, 22)
        pdf.set_text_color(180, 180, 210)
        pdf.cell(55, 7, f"  {label.capitalize()}", border=1, fill=True)
        pdf.cell(38, 7, f"{p*100:.1f}%",           border=1, fill=True)
        if ok:
            pdf.set_text_color(52, 211, 153)
        else:
            pdf.set_text_color(75, 85, 99)
        pdf.cell(32, 7, "DETECTED" if ok else "not present", border=1, fill=True)
        pdf.set_text_color(100, 100, 130)
        short = INST_DESC[label].split("-")[0].strip()
        pdf.cell(0, 7, short, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Chart ──
    sec("CONFIDENCE VISUALIZATION")
    chart_path = tempfile.mktemp(suffix='.png')
    with open(chart_path, 'wb') as f:
        f.write(chart_buf.getvalue())
    pdf.image(chart_path, x=15, w=175)
    os.unlink(chart_path)
    pdf.ln(4)

    # ── Spectrogram ──
    sec("WAVEFORM & MEL SPECTROGRAM")
    spec_path = tempfile.mktemp(suffix='.png')
    with open(spec_path, 'wb') as f:
        f.write(spec_buf.getvalue())
    pdf.image(spec_path, x=10, w=185)
    os.unlink(spec_path)
    pdf.ln(4)

    # ── Explanation ──
    sec("WHAT THESE RESULTS MEAN")
    pdf.set_font("Helvetica", "", 8); pdf.set_text_color(150, 150, 180)
    lines = [
        "InstruNet AI uses a CNN trained on mel-spectrograms from the NSynth dataset.",
        "Each 3-second segment is analyzed independently; confidence scores are averaged across all segments.",
        f"Instruments with confidence >= {threshold} are marked as DETECTED.",
        "Higher confidence = stronger presence of that instrument in the audio.",
        "Multi-label detection means multiple instruments can be detected simultaneously.",
    ]
    for line in lines:
        pdf.cell(0, 6, f"  * {line}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Timeline ──
    if timeline:
        pdf.add_page()
        sec("SEGMENT TIMELINE")
        pdf.set_fill_color(18, 16, 36); pdf.set_text_color(120, 110, 180)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(30, 7, "Start (s)",  border=1, fill=True)
        pdf.cell(30, 7, "End (s)",    border=1, fill=True)
        pdf.cell(0,  7, "Active Instruments", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        for seg in timeline:
            pdf.set_fill_color(12, 12, 22)
            pdf.set_text_color(150, 150, 180)
            pdf.cell(30, 6, str(seg["start"]), border=1, fill=True)
            pdf.cell(30, 6, str(seg["end"]),   border=1, fill=True)
            active = ", ".join([i.capitalize() for i in seg["instruments"]]) if seg["instruments"] else "-"
            if seg["instruments"]:
                pdf.set_text_color(52, 211, 153)
            else:
                pdf.set_text_color(75, 85, 99)
            pdf.cell(0, 6, active, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    # ── Footer ──
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 7); pdf.set_text_color(60, 60, 80)
    pdf.cell(0, 5, "Generated by InstruNet AI  |  CNN-Based Music Instrument Recognition  |  NSynth Dataset",
             align="C", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="app-header">
  <div>
    <div class="app-title">⬡ InstruNet AI</div>
    <div class="app-sub">CNN-Based Music Instrument Recognition</div>
  </div>
  <div class="app-version">v2.0 · NSynth · 9 Instruments</div>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error(f"Model failed to load: {model_err}")
    st.stop()

# Pipeline
pipeline_slot = st.empty()
pipeline_slot.markdown("""
<div class="pipeline">
  <div class="pipe-step active">① Input</div>
  <div class="pipe-arrow">›</div>
  <div class="pipe-step">② Processing</div>
  <div class="pipe-arrow">›</div>
  <div class="pipe-step">③ Insights</div>
  <div class="pipe-arrow">›</div>
  <div class="pipe-step">④ Actions</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
    st.markdown("---")
    threshold = st.slider("Detection Threshold", 0.10, 0.90, 0.25, 0.05,
                          help="Instruments above this confidence → detected")
    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Audio", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.65rem; color:#374151; line-height:2;'>
    SUPPORTED INSTRUMENTS<br>
    ◦ Keyboard &nbsp; ◦ Guitar<br>
    ◦ Bass &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◦ String<br>
    ◦ Brass &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◦ Reed<br>
    ◦ Mallet &nbsp;&nbsp;&nbsp;&nbsp; ◦ Organ<br>
    ◦ Flute
    </div>""", unsafe_allow_html=True)

# Main columns
col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.markdown('<div class="section-label">🎧 Audio Preview</div>', unsafe_allow_html=True)
    if uploaded_file:
        st.audio(uploaded_file)
    else:
        st.markdown('<div class="empty-box">↑ Upload a file from the sidebar</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="section-label">📊 Waveform & Spectrogram</div>', unsafe_allow_html=True)
    spec_slot = st.empty()
    spec_slot.markdown('<div class="empty-box">Appears after analysis</div>', unsafe_allow_html=True)

st.markdown("---")

# Placeholders
res_slot      = st.empty()
stats_slot    = st.empty()
timeline_slot = st.empty()
export_slot   = st.empty()

with res_slot.container():
    st.markdown('<div class="section-label">🎯 Prediction Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="empty-box">Results appear here after analysis</div>', unsafe_allow_html=True)


# ── Run analysis and store in session_state ──────────────────────────────────
if analyze_btn:
    if uploaded_file is None:
        st.warning("⚠️ Upload an audio file first.")
    else:
        pipeline_slot.markdown("""
        <div class="pipeline">
          <div class="pipe-step done">① Input ✓</div>
          <div class="pipe-arrow">›</div>
          <div class="pipe-step active">② Processing</div>
          <div class="pipe-arrow">›</div>
          <div class="pipe-step">③ Insights</div>
          <div class="pipe-arrow">›</div>
          <div class="pipe-step">④ Actions</div>
        </div>""", unsafe_allow_html=True)

        with st.spinner("Loading and preprocessing audio..."):
            suffix = ".mp3" if uploaded_file.name.lower().endswith(".mp3") else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                y_audio, sr_audio, duration = load_audio(tmp_path)
                spec_buf = make_vis(y_audio, sr_audio)
            except Exception as e:
                st.error(f"Audio load error: {e}"); os.unlink(tmp_path); st.stop()

        with st.spinner("Running CNN inference..."):
            try:
                avg_pred, timeline, num_segs = predict_segments(y_audio, sr_audio, threshold)
            except Exception as e:
                st.error(f"Prediction error: {e}"); os.unlink(tmp_path); st.stop()

        os.unlink(tmp_path)
        chart_buf = make_chart(avg_pred)

        # ✅ Save everything to session_state so download reruns don't lose results
        st.session_state['result'] = {
            'avg_pred':    avg_pred,
            'timeline':    timeline,
            'num_segs':    num_segs,
            'duration':    duration,
            'spec_buf':    spec_buf,
            'chart_buf':   chart_buf,
            'filename':    uploaded_file.name,
            'threshold':   threshold,
        }

# ── Render results from session_state (survives download button reruns) ───────
if 'result' in st.session_state:
    r          = st.session_state['result']
    avg_pred   = r['avg_pred']
    timeline   = r['timeline']
    num_segs   = r['num_segs']
    duration   = r['duration']
    spec_buf   = r['spec_buf']
    chart_buf  = r['chart_buf']
    filename   = r['filename']
    threshold  = r['threshold']
    detected_instruments = [LABELS[i] for i in range(len(LABELS)) if avg_pred[i] >= threshold]

    pipeline_slot.markdown("""
    <div class="pipeline">
      <div class="pipe-step done">① Input ✓</div>
      <div class="pipe-arrow">›</div>
      <div class="pipe-step done">② Processing ✓</div>
      <div class="pipe-arrow">›</div>
      <div class="pipe-step done">③ Insights ✓</div>
      <div class="pipe-arrow">›</div>
      <div class="pipe-step active">④ Actions</div>
    </div>""", unsafe_allow_html=True)

    # Spectrogram
    spec_slot.image(spec_buf, use_container_width=True)

    # Results
    with res_slot.container():
        st.markdown('<div class="section-label">🎯 Prediction Results</div>', unsafe_allow_html=True)

        if detected_instruments:
            chips = "".join([
                f'<span class="chip" style="background:{CHIP_COLORS[inst][0]};color:{CHIP_COLORS[inst][1]};border-color:{CHIP_COLORS[inst][1]}40;">{inst}</span>'
                for inst in detected_instruments
            ])
            st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#4b5563;font-size:0.8rem;">No instruments detected above threshold.</p>',
                        unsafe_allow_html=True)

        # ✅ Build bars as one string — never call st.markdown inside a list comprehension
        bars_html = ""
        for idx, lbl in enumerate(LABELS):
            pct   = float(avg_pred[idx]) * 100
            color = COLORS[lbl]
            badge = '<span class="inst-badge">yes</span>' if avg_pred[idx] >= threshold else '<span class="inst-badge-no">-</span>'
            bars_html += f"""
            <div class="inst-row">
              <span class="inst-name">{lbl}</span>
              <div class="bar-track">
                <div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
              </div>
              <span class="inst-pct">{pct:.0f}%</span>
              {badge}
            </div>"""
        st.markdown(f'<div class="card">{bars_html}</div>', unsafe_allow_html=True)

    # Stats
    with stats_slot.container():
        st.markdown('<div class="section-label">📈 Insights</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        top_i = LABELS[int(np.argmax(avg_pred))]
        top_c = float(np.max(avg_pred)) * 100

        # ✅ Build each stat box as a string — never call st.markdown inside columns comprehension
        with s1:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{len(detected_instruments)}</div><div class="stat-lbl">Detected</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-box"><div class="stat-num" style="font-size:1.1rem;padding-top:.5rem;">{top_i}</div><div class="stat-lbl">Top Instrument</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{top_c:.0f}%</div><div class="stat-lbl">Top Confidence</div></div>', unsafe_allow_html=True)
        with s4:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{round(duration,1)}s</div><div class="stat-lbl">Duration</div></div>', unsafe_allow_html=True)

    # Timeline
    if timeline:
        with timeline_slot.container():
            st.markdown("---")
            st.markdown('<div class="section-label">🕐 Segment Timeline</div>', unsafe_allow_html=True)

            # ✅ Build full table HTML as one string first
            rows_html = ""
            for seg in timeline:
                insts = ", ".join(seg['instruments']) if seg['instruments'] else "none"
                color = "#34d399" if seg['instruments'] else "#374151"
                rows_html += f"""
                <div class="tl-row">
                  <span class="tl-time">{seg['start']}s to {seg['end']}s</span>
                  <span style="color:{color}; font-size:0.78rem;">{insts}</span>
                </div>"""
            st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

    # Export
    with export_slot.container():
        st.markdown("---")
        st.markdown('<div class="section-label">📁 Export Actions</div>', unsafe_allow_html=True)

        base      = os.path.splitext(filename)[0]
        json_data = build_json(filename, duration, avg_pred, timeline, threshold, num_segs)
        pdf_data  = build_pdf(filename, duration, avg_pred, timeline, threshold, num_segs, spec_buf, chart_buf)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download JSON Report",
                data=json_data,
                file_name=f"{base}_instrunet.json",
                mime="application/json",
                use_container_width=True
            )
        with d2:
            st.download_button(
                "Download PDF Report",
                data=pdf_data,
                file_name=f"{base}_instrunet.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        st.markdown("""
        <p style='font-family:Space Mono,monospace;font-size:0.62rem;color:#374151;margin-top:0.5rem;'>
        JSON: metadata + model config + per-instrument scores + timeline &nbsp;|&nbsp;
        PDF: full report with spectrogram + confidence chart + timeline table
        </p>""", unsafe_allow_html=True)