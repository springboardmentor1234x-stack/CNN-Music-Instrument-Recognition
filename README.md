# InstruNet AI – CNN-Based Music Instrument Recognition System

An end-to-end AI project that detects musical instruments in audio tracks using deep learning (CNNs on Mel-Spectrograms).

## Features
- **Upload Audio**: Supports WAV and MP3 files.
- **Preprocessing**: Automatically converts audio to mono, normalizes, and extracts Mel-Spectrogram features.
- **Time-Segmented Prediction**: Analyzes the audio in 3-second chunks to provide a timeline of instrument intensity.
- **Visualizations**: Displays waveforms, spectrograms, and an interactive instrument confidence timeline.
- **Exporting**: Download JSON or PDF reports detailing the findings.

## Project Structure
- `data/` : Raw and synthetic datasets.
- `preprocessing/` : Tools for audio standardization and spectrogram feature extraction.
- `models/` : CNN definitions and saved network weights (`.h5`).
- `training/` : Scripts to generate data and train the AI model.
- `app/` : Streamlit frontend interface.
- `utils/` : Inference functions and export handlers.

## How to Run

1. **Activate Environment** (If applicable):
   Ensure you have your Python virtual environment active.

2. **Run the Application**:
   Execute the following command in the root project directory:
   ```bash
   streamlit run app/main.py
   ```
   A browser window will open automatically with the InstruNet AI interface.

3. **Using the App**:
   - Upload any `.wav` or `.mp3` file.
   - Click **Analyze Audio**.
   - Review predictions, the timeline graphic, and download the associated JSON or PDF reports!

*Note: For this template, a small synthetic dataset. To train on a real dataset like NSynth, replace the data folder contents, update the `training/train.py` class mappings, and re-run training.*
