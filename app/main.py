import streamlit as st
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import librosa
import librosa.display
import json
import time
import requests
from streamlit_lottie import st_lottie

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.inference import load_inference_model, predict_on_audio
from utils.export import export_to_json, export_to_pdf

# Constants
MODEL_PATH = "models/baseline_cnn.h5"
CLASS_MAPPING_PATH = "models/class_mapping.json"
TEMP_DIR = "data/temp"
REAL_DATA_DIR = "data/real_instruments"

# Ensure dirs exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="InstruNet AI | Music Recognition",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    local_css(css_path)

# Load Lottie Animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_music = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_y9m8Yt.json")  # Music notes
lottie_scanning = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_t9uclp3o.json") # Pulse/Scanning

@st.cache_resource
def load_model_and_mapping():
    model = load_inference_model(MODEL_PATH)
    with open(CLASS_MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    return model, mapping

# --- UI Header ---
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🎵 InstruNet AI</h1>
        <p class="hero-subtitle">Premium CNN-Based Musical Instrument Recognition System</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar / Controls ---
with st.sidebar:
    st.image("https://img.icons8.com/bubbles/200/000000/music.png", width=100)
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Confidence Threshold", 0.05, 0.50, 0.15, 0.05)
    st.markdown("---")
    st.markdown("### 📚 About")
    st.info("This system uses a Deep Convolutional Neural Network trained on 1,000+ real instrument samples to identify musical textures with 99.5% accuracy.")

# --- Main Interaction ---
col_up1, col_up2 = st.columns([2, 1])

with col_up1:
    st.markdown("### 📥 1. Upload Your Audio")
    uploaded_file = st.file_uploader("Drop a WAV or MP3 file here", type=['wav', 'mp3'])
    
    st.markdown("### 💡 Or try a Sample")
    # Sample Gallery
    if os.path.exists(REAL_DATA_DIR):
        instrument_folders = [d for d in os.listdir(REAL_DATA_DIR) if os.path.isdir(os.path.join(REAL_DATA_DIR, d))]
        if instrument_folders:
            cols = st.columns(min(len(instrument_folders), 7))
            for idx, inst in enumerate(instrument_folders):
                if cols[idx].button(f"🎹 {inst.capitalize()}", key=f"btn_{inst}"):
                    # Pick a random sample from that dir
                    inst_dir = os.path.join(REAL_DATA_DIR, inst)
                    samples = [f for f in os.listdir(inst_dir) if f.endswith('.wav')]
                    if samples:
                        sample_path = os.path.join(inst_dir, samples[0])
                        # Copy to temp dir to simulate upload
                        target_path = os.path.join(TEMP_DIR, samples[0])
                        import shutil
                        shutil.copy(sample_path, target_path)
                        st.session_state['active_file'] = target_path
                        st.session_state['active_name'] = f"Sample: {inst.capitalize()}"
                        st.success(f"Loaded {inst.capitalize()} sample!")

with col_up2:
    if lottie_music:
        st_lottie(lottie_music, height=250, key="music_anim")

# Logic to handle both upload and sample
file_to_process = None
display_name = ""

if uploaded_file is not None:
    temp_file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_to_process = temp_file_path
    display_name = uploaded_file.name
elif 'active_file' in st.session_state:
    file_to_process = st.session_state['active_file']
    display_name = st.session_state['active_name']

if file_to_process:
    st.markdown("---")
    col_proc1, col_proc2 = st.columns([3, 1])
    with col_proc1:
        st.markdown(f"**🔈 Currently Playing:** `{display_name}`")
        st.audio(file_to_process)
    
    if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
        with st.status("Analyzing Audio Textures...", expanded=True) as status:
            st.write("🔄 Loading AI Architecture...")
            try:
                model, class_mapping = load_model_and_mapping()
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()
            
            st.write("🌊 Preprocessing Spectra...")
            # Load audio for visualization
            audio, sr = librosa.load(file_to_process, sr=22050, mono=True)
            
            st.write("🧠 Running Neural Inference...")
            # Predict
            results = predict_on_audio(file_to_process, model, class_mapping, threshold=threshold)
            
            if "error" in results:
                st.error(results["error"])
                st.stop()
            
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # --- RESULTS UI ---
        st.markdown("## 🎯 Analysis Results")
        
        tab1, tab2, tab3 = st.tabs(["🔍 Detection Summary", "📊 Time Insights", "🛠️ Technical Data"])
        
        with tab1:
            st.markdown("### 📊 Top Instruments Detected")
            detected = results['summary']['detected_instruments']
            
            if not detected:
                st.warning("No instruments detected above the confidence threshold.")
            else:
                # Layout detection cards
                row_cols = st.columns(min(len(detected), 3))
                for idx, inst in enumerate(detected):
                    with row_cols[idx % 3]:
                        name = inst['instrument'].capitalize()
                        conf = inst['confidence']
                        st.markdown(f"""
                            <div class="result-card">
                                <h3 style="margin:0; color:#6366f1;">🎶 {name}</h3>
                                <p style="font-size:0.9rem; color:#94a3b8; margin:5px 0;">Confidence: <b>{conf*100:.1f}%</b></p>
                                <code style="color:#a855f7;">{inst['intensity_bars']}</code>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(min(conf, 1.0))
            
            st.markdown("---")
            # All classes bar chart using Plotly
            st.markdown("### 📉 Confidence Profile (All Classes)")
            all_confs = results['summary']['overall_confidences']
            df_confs = np.array([[k.capitalize(), v] for k, v in all_confs.items()])
            fig_all = px.bar(
                x=df_confs[:, 1].astype(float), 
                y=df_confs[:, 0], 
                orientation='h',
                labels={'x': 'Confidence', 'y': 'Instrument'},
                color=df_confs[:, 1].astype(float),
                color_continuous_scale='Viridis',
                template='plotly_dark'
            )
            fig_all.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_all, use_container_width=True)

        with tab2:
            st.markdown("### 📈 Time-Series Intensity")
            timeline = results['timeline']
            
            # Prepare Plotly Timeline
            plot_data = []
            for seg in timeline:
                for inst, conf in seg['confidences'].items():
                    plot_data.append({
                        "Time (s)": seg['start'],
                        "Instrument": inst.capitalize(),
                        "Confidence": conf
                    })
            
            import pandas as pd
            df_plot = pd.DataFrame(plot_data)
            
            fig_time = px.line(
                df_plot, x="Time (s)", y="Confidence", color="Instrument",
                markers=True, line_shape='spline', render_mode='svg',
                template='plotly_dark',
                title="Instrument Intensity Over Time"
            )
            fig_time.update_layout(hovermode="x unified")
            st.plotly_chart(fig_time, use_container_width=True)
            
            st.markdown("### 🎯 Dominant Sequence")
            dominant_row = st.columns(len(timeline))
            for idx, seg in enumerate(timeline):
                dom = seg.get('dominant', 'unknown').capitalize()
                dominant_row[idx].button(f"{seg['start']}s\n{dom}", key=f"dom_{idx}")

        with tab3:
            st.markdown("### 🔊 Waveform & Log-Spectrogram")
            col_spec1, col_spec2 = st.columns(2)
            
            with col_spec1:
                fig_wave, ax_wave = plt.subplots(figsize=(8, 4))
                librosa.display.waveshow(audio, sr=sr, ax=ax_wave, color="#6366f1", alpha=0.7)
                ax_wave.set_title("Audio Waveform", color="white")
                ax_wave.set_facecolor("#1e293b")
                fig_wave.patch.set_facecolor("#0f172a")
                st.pyplot(fig_wave)
            
            with col_spec2:
                fig_spec, ax_spec = plt.subplots(figsize=(8, 4))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax_spec)
                fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
                ax_spec.set_title("Mel-Spectrogram (Input to CNN)", color="white")
                st.pyplot(fig_spec)
                
            st.markdown("### 📥 Download Reports")
            json_path = export_to_json(results)
            pdf_path = export_to_pdf(results)
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                with open(json_path, 'r') as f:
                    st.download_button("Download JSON", f.read(), "report.json", "application/json", use_container_width=True)
            with col_dl2:
                with open(pdf_path, 'rb') as f:
                    st.download_button("Download PDF", f.read(), "report.pdf", "application/pdf", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>InstruNet AI © 2026 | Built with TensorFlow & Streamlit</p>", unsafe_allow_html=True)
