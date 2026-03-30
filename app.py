# app.py
import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import io
import tempfile
from pipeline import run_pipeline

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .stMarkdown p { color: #ffffff; }
    .stMarkdown h1 { color: #00d4ff; }
    .stMarkdown h2 { color: #ffffff; }
    .stMarkdown h3 { color: #ffffff; }
    .instrument-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #ffffff;
        border-left: 4px solid #00d4ff;
    }
    .detected {
        border-left: 4px solid #00ff88 !important;
    }
    .not-detected {
        border-left: 4px solid #444444 !important;
        opacity: 0.6;
    }
    .section-header {
        background: linear-gradient(90deg, #1e2130, #0e1117);
        border-left: 5px solid #00d4ff;
        border-radius: 8px;
        padding: 12px 20px;
        margin: 15px 0;
        color: #00d4ff;
        font-size: 20px;
        font-weight: bold;
    }
    .lock-box {
        background: #1e2130;
        border: 2px dashed #444444;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        color: #888888;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── CONSTANTS ─────────────────────────────────────────────────────
CLASS_NAMES = [
    'cel', 'cla', 'flu', 'gac', 'gel',
    'org', 'pia', 'sax', 'tru', 'vio', 'voi'
]

INSTRUMENT_FULL_NAMES = {
    'cel': 'Cello',
    'cla': 'Clarinet',
    'flu': 'Flute',
    'gac': 'Acoustic Guitar',
    'gel': 'Electric Guitar',
    'org': 'Organ',
    'pia': 'Piano',
    'sax': 'Saxophone',
    'tru': 'Trumpet',
    'vio': 'Violin',
    'voi': 'Voice'
}


# ── LOAD MODEL ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "instrunet_model_best.keras",
            compile=False
        )
        return model
    except Exception:
        try:
            model = tf.keras.models.load_model(
                "instrunet_model_best.h5",
                compile=False
            )
            return model
        except Exception as e:
            st.error(f"Cannot load model: {e}")
            return None


# ── JSON EXPORT ───────────────────────────────────────────────────
def get_json_export(report):
    export = {
        "metadata": {
            "audio_name"        : report["metadata"]["audio_name"],
            "audio_duration_sec": report["metadata"]["audio_duration_sec"],
            "report_generated"  : report["metadata"]["report_generated"]
        },
        "model_config": {
            "threshold"           : report["model_config"]["threshold"],
            "segment_duration_sec": report["model_config"]["segment_duration_sec"],
            "hop_duration_sec"    : report["model_config"]["hop_duration_sec"],
            "total_segments"      : report["model_config"]["total_segments"],
            "aggregation_method"  : "mean"
        },
        "predictions": {
            "detected_instruments": report["predictions"]["detected_instruments"],
            "instrument_wise"     : report["predictions"]["instrument_wise"]
        },
        "confidence": report["confidence"],
        "timelines" : report["timelines"]
    }
    return json.dumps(export, indent=4)


# ── VISUALIZATION IMAGE FOR PDF ───────────────────────────────────
def generate_viz_image(y_vis, sr_vis, mean_probs,
                       all_segment_probs, detected,
                       threshold, hop_duration,
                       segment_duration):

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#1a1a2e')

    # Waveform
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.set_facecolor('#16213e')
    times = np.linspace(0, len(y_vis)/sr_vis, len(y_vis))
    ax1.plot(times, y_vis,
             color='#00d4ff', linewidth=0.5, alpha=0.8)
    ax1.set_title('Waveform', color='white', fontsize=11)
    ax1.set_xlabel('Time (s)', color='#aaaaaa', fontsize=9)
    ax1.set_ylabel('Amplitude', color='#aaaaaa', fontsize=9)
    ax1.tick_params(colors='#aaaaaa')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444444')

    # Mel Spectrogram
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.set_facecolor('#16213e')
    mel    = librosa.feature.melspectrogram(
        y=y_vis, sr=sr_vis,
        n_fft=2048, hop_length=512, n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(
        mel_db, sr=sr_vis, hop_length=512,
        x_axis='time', y_axis='mel',
        cmap='magma', ax=ax2
    )
    ax2.set_title('Mel Spectrogram', color='white', fontsize=11)
    ax2.set_xlabel('Time (s)', color='#aaaaaa', fontsize=9)
    ax2.tick_params(colors='#aaaaaa')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444444')

    # Intensity bars
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.set_facecolor('#16213e')
    names  = [INSTRUMENT_FULL_NAMES.get(n, n) for n in CLASS_NAMES]
    probs  = [float(p) for p in mean_probs]
    colors = ['#00ff88' if CLASS_NAMES[i] in detected
              else '#444444'
              for i in range(len(CLASS_NAMES))]
    y_pos  = np.arange(len(names))
    bars   = ax3.barh(y_pos, probs, color=colors,
                      height=0.6, alpha=0.85)
    ax3.axvline(x=threshold, color='#ff4444',
                linestyle='--', linewidth=1.5)
    for bar, prob in zip(bars, probs):
        ax3.text(prob + 0.005,
                 bar.get_y() + bar.get_height()/2,
                 f'{prob:.3f}',
                 va='center', color='white', fontsize=8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, color='white', fontsize=9)
    ax3.set_xlim(0, 1.0)
    ax3.set_xlabel('Probability', color='#aaaaaa')
    ax3.tick_params(colors='#aaaaaa')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#444444')

    # Timeline
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.set_facecolor('#16213e')
    segment_probs = np.array(all_segment_probs)
    time_points   = [
        i * hop_duration + segment_duration / 2
        for i in range(len(segment_probs))
    ]
    colors_t = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    for i, name in enumerate(CLASS_NAMES):
        if name in detected:
            ax4.plot(
                time_points,
                segment_probs[:, i],
                label=INSTRUMENT_FULL_NAMES.get(name, name),
                color=colors_t[i],
                linewidth=2, alpha=0.85
            )
    ax4.axhline(y=threshold, color='#ff4444',
                linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Time (s)', color='#aaaaaa', fontsize=10)
    ax4.set_ylabel('Probability', color='#aaaaaa', fontsize=10)
    ax4.set_ylim(0, 1.0)
    ax4.tick_params(colors='#aaaaaa')
    ax4.legend(facecolor='#16213e', labelcolor='white',
               fontsize=9, loc='upper right', ncol=3)
    for spine in ax4.spines.values():
        spine.set_edgecolor('#444444')

    plt.suptitle('InstruNet AI Analysis Report',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    buf.seek(0)
    return buf


# ── PDF EXPORT ────────────────────────────────────────────────────
def generate_pdf(report, viz_buf=None):
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()

        # Project title
        pdf.set_fill_color(26, 26, 46)
        pdf.rect(0, 0, 210, 40, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_xy(10, 8)
        pdf.cell(0, 10, 'InstruNet AI', ln=True)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_xy(10, 20)
        pdf.cell(0, 8,
                 'Music Instrument Recognition Report',
                 ln=True)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_xy(10, 30)
        pdf.cell(0, 6,
                 f'Generated: {report["metadata"]["report_generated"]}',
                 ln=True)

        # Audio file details
        pdf.set_text_color(0, 0, 0)
        pdf.set_xy(10, 48)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Audio File Details', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6,
                 f'File     : {report["metadata"]["audio_name"]}',
                 ln=True)
        pdf.cell(0, 6,
                 f'Duration : {report["metadata"]["audio_duration_sec"]}s',
                 ln=True)
        pdf.cell(0, 6,
                 f'Segments : {report["model_config"]["total_segments"]}',
                 ln=True)
        pdf.cell(0, 6,
                 f'Threshold: {report["model_config"]["threshold"]}',
                 ln=True)

        # Short explanation
        pdf.ln(3)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(
            0, 5,
            'InstruNet AI uses a CNN trained on mel spectrograms '
            'to identify musical instruments in audio tracks. '
            'Audio is split into segments, each analyzed separately, '
            'and predictions are averaged to produce final results.'
        )

        # Detected instruments summary
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Detected Instruments Summary', ln=True)
        pdf.set_font('Helvetica', '', 10)

        detected = report['predictions']['detected_instruments']
        if detected:
            for name in detected:
                full = INSTRUMENT_FULL_NAMES.get(name, name)
                prob = report['confidence'][name]
                pdf.set_text_color(0, 140, 0)
                pdf.cell(
                    0, 7,
                    f'  + {full} - Present - {prob:.1%}',
                    ln=True
                )
        else:
            pdf.set_text_color(160, 0, 0)
            pdf.cell(0, 7, '  No instruments detected', ln=True)

        # Confidence visualization
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8,
                 'Confidence Visualization - All Instruments',
                 ln=True)
        pdf.set_font('Helvetica', '', 10)

        for name, score in sorted(
            report['confidence'].items(),
            key=lambda x: -x[1]
        ):
            full = INSTRUMENT_FULL_NAMES.get(name, name)
            stat = report['predictions']['instrument_wise'][name]['status']
            bar  = '|' * int(score * 30)

            if stat == 'Present':
                pdf.set_text_color(0, 140, 0)
            else:
                pdf.set_text_color(160, 0, 0)

            pdf.cell(
                0, 6,
                f'{full:<20} {score:.3f} {bar}',
                ln=True
            )

        # Optional waveform/spectrogram images
        if viz_buf is not None:
            pdf.add_page()
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8,
                     'Waveform and Spectrogram Analysis',
                     ln=True)
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(
                0, 5,
                'Visual analysis including waveform, mel spectrogram, '
                'instrument intensity bars and activity timeline.'
            )
            with tempfile.NamedTemporaryFile(
                delete=False, suffix='.png'
            ) as tmp_img:
                tmp_img.write(viz_buf.getvalue())
                tmp_img_path = tmp_img.name
            pdf.image(tmp_img_path, x=10, w=190)
            os.unlink(tmp_img_path)

        pdf_bytes = bytes(pdf.output())
        return pdf_bytes

    except Exception as e:
        st.error(f"PDF Error: {e}")
        print(f"PDF Error: {e}")
        return None


# ── HEADER ────────────────────────────────────────────────────────
st.markdown("# InstruNet AI")
st.markdown("### CNN-Based Music Instrument Recognition System")
st.markdown("*Upload. Analyze. Discover.*")
st.markdown("---")


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    threshold = st.slider(
        "Detection Threshold", 0.05, 0.50, 0.15, 0.01,
        help="Lower = more instruments detected"
    )
    segment_duration = st.slider(
        "Segment Duration (s)", 1.0, 5.0, 3.0, 0.5
    )
    hop_duration = st.slider(
        "Hop Duration (s)", 0.5, 3.0, 1.5, 0.5
    )
    st.markdown("---")
    st.markdown("### Supported Instruments")
    for k, v in INSTRUMENT_FULL_NAMES.items():
        st.markdown(f"- {v}")


# ── MAIN LAYOUT ───────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("## Upload Audio")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac'],
        help="Upload WAV, MP3 or FLAC"
    )

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        st.success(f"Loaded: {uploaded_file.name}")
        st.info(f"Size: {uploaded_file.size / 1024:.1f} KB")
        run_btn = st.button(
            "Analyze Track",
            type="primary",
            use_container_width=True
        )
    else:
        st.info("Upload a WAV/MP3/FLAC to get started")
        run_btn = False

with col2:
    st.markdown("## Analysis Results")
    if not uploaded_file:
        st.markdown("""
        <div class="lock-box">
            <br>
            <b>Results will appear here</b><br><br>
            Upload an audio file and click
            Analyze Track to unlock results
        </div>
        """, unsafe_allow_html=True)
    elif uploaded_file and not run_btn:
        st.markdown("""
        <div class="lock-box">
            <br>
            <b>File loaded successfully!</b><br><br>
            Click Analyze Track to start analysis
        </div>
        """, unsafe_allow_html=True)


# ── RUN ANALYSIS ──────────────────────────────────────────────────
if uploaded_file and run_btn:

    with col2:
        with st.spinner("Analyzing audio track..."):

            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            model = load_model()
            if model is None:
                st.stop()

            report, mean_probs, all_segment_probs = run_pipeline(
                tmp_path, model, CLASS_NAMES,
                segment_duration=segment_duration,
                hop_duration=hop_duration,
                threshold=threshold
            )
            os.unlink(tmp_path)

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp2:
                uploaded_file.seek(0)
                tmp2.write(uploaded_file.read())
                tmp2_path = tmp2.name
            y_vis, sr_vis = librosa.load(tmp2_path, sr=22050)
            os.unlink(tmp2_path)

        # ── SUCCESS POPUP ─────────────────────────────────────────
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #0d3b1f, #1a1a2e);
            border: 3px solid #00ff88;
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            margin: 10px 0;
        ">
            <div style="font-size: 48px;">✅</div>
            <div style="
                color: #00ff88;
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            ">
                Track Successfully Analyzed!
            </div>
            <div style="color: #aaaaaa; font-size: 14px;">
                Scroll down to view your Prediction Results
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── METRICS ───────────────────────────────────────────────
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Duration",
                      f"{report['metadata']['audio_duration_sec']}s")
        with m2:
            st.metric("Segments",
                      report['model_config']['total_segments'])
        with m3:
            st.metric("Detected",
                      len(report['predictions']['detected_instruments']))
        with m4:
            st.metric("Threshold",
                      report['model_config']['threshold'])

        st.markdown("---")

        # ── PREDICTION RESULTS ────────────────────────────────────
        st.markdown("""
        <div class="section-header">
            Prediction Results
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1])

        with r1:
            st.markdown("#### Detected Instrument Summary")
            detected = report['predictions']['detected_instruments']

            if detected:
                for name in detected:
                    full = INSTRUMENT_FULL_NAMES.get(name, name)
                    prob = report['confidence'][name]
                    st.markdown(f"""
                    <div class="instrument-card detected">
                        <b>{full}</b>
                        <span style="
                            float:right;
                            color:#00ff88;
                            font-weight:bold;
                        ">{prob:.1%}</span>
                        <div style="
                            background:#333;
                            border-radius:4px;
                            margin-top:6px;
                            height:8px;
                        ">
                            <div style="
                                background:#00ff88;
                                width:{prob*100:.1f}%;
                                height:8px;
                                border-radius:4px;
                            "></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(
                    "No instruments detected above threshold. "
                    "Try lowering the threshold in sidebar."
                )

        with r2:
            st.markdown("#### Confidence Visualization")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            fig1.patch.set_facecolor('#0e1117')
            ax1.set_facecolor('#1e2130')

            names  = [INSTRUMENT_FULL_NAMES.get(n, n)
                      for n in CLASS_NAMES]
            probs  = [report['confidence'][n] for n in CLASS_NAMES]
            colors = [
                '#00ff88'
                if CLASS_NAMES[i] in detected
                else '#444444'
                for i in range(len(CLASS_NAMES))
            ]
            y_pos = np.arange(len(names))
            bars  = ax1.barh(y_pos, probs, color=colors,
                             height=0.6, alpha=0.85)
            ax1.axvline(x=threshold, color='#ff4444',
                        linestyle='--', linewidth=1.5,
                        label=f'Threshold ({threshold})')
            for bar, prob in zip(bars, probs):
                ax1.text(
                    prob + 0.005,
                    bar.get_y() + bar.get_height()/2,
                    f'{prob:.3f}',
                    va='center', color='white', fontsize=8
                )
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(names, color='white', fontsize=9)
            ax1.set_xlim(0, 1.0)
            ax1.set_xlabel('Probability', color='#aaaaaa')
            ax1.tick_params(colors='#aaaaaa')
            ax1.legend(facecolor='#1e2130',
                       labelcolor='white', fontsize=8)
            for spine in ax1.spines.values():
                spine.set_edgecolor('#444444')
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

        st.markdown("---")

        # ── AUDIO VISUALIZATION ───────────────────────────────────
        st.markdown("""
        <div class="section-header">
            Audio Visualization
        </div>
        """, unsafe_allow_html=True)

        v1, v2 = st.columns(2)

        with v1:
            st.markdown("**Waveform**")
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            fig2.patch.set_facecolor('#0e1117')
            ax2.set_facecolor('#1e2130')
            times = np.linspace(
                0, len(y_vis)/sr_vis, len(y_vis)
            )
            ax2.plot(times, y_vis,
                     color='#00d4ff', linewidth=0.5, alpha=0.8)
            ax2.set_title('Waveform', color='white', fontsize=11)
            ax2.set_xlabel('Time (s)', color='#aaaaaa', fontsize=9)
            ax2.set_ylabel('Amplitude', color='#aaaaaa', fontsize=9)
            ax2.tick_params(colors='#aaaaaa')
            for spine in ax2.spines.values():
                spine.set_edgecolor('#444444')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        with v2:
            st.markdown("**Mel Spectrogram**")
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            fig3.patch.set_facecolor('#0e1117')
            ax3.set_facecolor('#1e2130')
            mel    = librosa.feature.melspectrogram(
                y=y_vis, sr=sr_vis,
                n_fft=2048, hop_length=512, n_mels=128
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            librosa.display.specshow(
                mel_db, sr=sr_vis, hop_length=512,
                x_axis='time', y_axis='mel',
                cmap='magma', ax=ax3
            )
            ax3.set_title('Mel Spectrogram',
                          color='white', fontsize=11)
            ax3.set_xlabel('Time (s)', color='#aaaaaa', fontsize=9)
            ax3.tick_params(colors='#aaaaaa')
            for spine in ax3.spines.values():
                spine.set_edgecolor('#444444')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        st.markdown("---")

        # ── TIMELINE ──────────────────────────────────────────────
        st.markdown("""
        <div class="section-header">
            Instrument Activity Timeline
        </div>
        """, unsafe_allow_html=True)

        segment_probs = np.array(all_segment_probs)
        time_points   = [
            i * hop_duration + segment_duration / 2
            for i in range(len(segment_probs))
        ]
        colors_t = plt.cm.tab10(
            np.linspace(0, 1, len(CLASS_NAMES))
        )

        fig4, ax4 = plt.subplots(figsize=(14, 5))
        fig4.patch.set_facecolor('#0e1117')
        ax4.set_facecolor('#1e2130')

        for i, name in enumerate(CLASS_NAMES):
            if name in detected:
                ax4.plot(
                    time_points,
                    segment_probs[:, i],
                    label=INSTRUMENT_FULL_NAMES.get(name, name),
                    color=colors_t[i],
                    linewidth=2, alpha=0.85
                )

        ax4.axhline(y=threshold, color='#ff4444',
                    linestyle='--', linewidth=1.5,
                    label=f'Threshold ({threshold})')
        ax4.set_xlabel('Time (seconds)',
                       color='#aaaaaa', fontsize=10)
        ax4.set_ylabel('Probability',
                       color='#aaaaaa', fontsize=10)
        ax4.set_ylim(0, 1.0)
        ax4.tick_params(colors='#aaaaaa')
        ax4.legend(facecolor='#1e2130', labelcolor='white',
                   fontsize=9, loc='upper right', ncol=3)
        for spine in ax4.spines.values():
            spine.set_edgecolor('#444444')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        st.markdown("---")

        # ── OPTIONAL DETAILS ──────────────────────────────────────
        st.markdown("""
        <div class="section-header">
            Optional Details
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View Full JSON Report", expanded=False):
            st.json(report)

        with st.expander("View Instrument Table", expanded=False):
            import pandas as pd
            table_data = [
                {
                    "Instrument": INSTRUMENT_FULL_NAMES.get(n, n),
                    "Confidence": f"{report['confidence'][n]:.3f}",
                    "Percentage": f"{report['confidence'][n]:.1%}",
                    "Status"    : report['predictions']
                                  ['instrument_wise'][n]['status']
                }
                for n in sorted(
                    CLASS_NAMES,
                    key=lambda x: -report['confidence'][x]
                )
            ]
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

        st.markdown("---")

        # ── EXPORT ────────────────────────────────────────────────
        st.markdown("""
        <div class="section-header">
            Export Report
        </div>
        """, unsafe_allow_html=True)

        viz_buf = generate_viz_image(
            y_vis, sr_vis, mean_probs,
            all_segment_probs,
            report['predictions']['detected_instruments'],
            threshold, hop_duration, segment_duration
        )

        e1, e2 = st.columns(2)

        with e1:
            json_str = get_json_export(report)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}"
                          f"_instruments.json",
                mime="application/json",
                use_container_width=True
            )

        with e2:
            pdf_bytes = generate_pdf(report, viz_buf)
            if pdf_bytes:
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}"
                              f"_instruments.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            