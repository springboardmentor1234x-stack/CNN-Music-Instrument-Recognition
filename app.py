import streamlit as st
import os
from src.pipeline import run_pipeline
from src.predict import save_json
from src.report import generate_pdf

st.set_page_config(page_title="InstruNet AI", page_icon="🎵", layout="centered")

# 🌸 LIGHT PURPLE PROFESSIONAL THEME
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #f5f3ff, #ede9fe, #ddd6fe);
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #5b21b6;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6d28d9;
    margin-bottom: 25px;
}

/* Buttons */
.stButton>button {
    background: #7c3aed;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    border: none;
    font-weight: 500;
}

.stButton>button:hover {
    background: #5b21b6;
}

/* Upload box */
.stFileUploader {
    border: 2px dashed #a78bfa;
    border-radius: 10px;
    padding: 10px;
}

/* Progress */
.stProgress > div > div > div > div {
    background-color: #7c3aed;
}

</style>
""", unsafe_allow_html=True)

# 🎵 HEADER
st.markdown("<div class='title'>🎵 InstruNet AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Musical Instrument Detection System</div>", unsafe_allow_html=True)

st.divider()

# 📂 UPLOAD SECTION
st.subheader("📂 Upload Audio File")
file = st.file_uploader("Upload a .wav file", type=["wav"])

if file:
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", "input.wav")

    with open(filepath, "wb") as f:
        f.write(file.read())

    st.success("File uploaded successfully")
    st.audio(filepath)

    st.divider()

    if st.button("🚀 Run Detection"):
        with st.spinner("Processing audio..."):

            result = run_pipeline(filepath)

            os.makedirs("outputs", exist_ok=True)
            save_json(result)
            generate_pdf(result)

        st.success("Detection Complete")

        st.divider()

        st.subheader("🎯 Detected Instruments")

        if len(result) == 0:
            st.warning("No instruments detected")
        else:
            for instrument, score in result.items():
                st.write(f"🎵 {instrument.capitalize()}")
                st.progress(score)

        st.divider()

        st.subheader("📥 Download Results")

        with open("outputs/result.json", "rb") as f:
            st.download_button("Download JSON", f, file_name="result.json")

        with open("outputs/report.pdf", "rb") as f:
            st.download_button("Download PDF", f, file_name="report.pdf")