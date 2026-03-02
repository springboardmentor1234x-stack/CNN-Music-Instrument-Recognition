# InstruNet AI

### CNN-Based Musical Instrument Recognition System

InstruNet AI is a deep learning–based system that automatically identifies musical instruments present in an audio file.
Instead of manually listening and tagging instruments, this system converts audio into mel-spectrogram images and uses a Convolutional Neural Network (CNN) to detect instruments such as piano, guitar, and violin.

The project demonstrates a complete end-to-end machine learning pipeline, including data preprocessing, model training, prediction, visualization, and report generation.

---

## Features

* Automatic instrument detection from audio files
* CNN-based spectrogram classification
* Audio waveform visualization
* Confidence score graph
* Instrument intensity visualization
* JSON and PDF report generation
* Interactive web interface using Streamlit

---

## Instruments Supported

The current trained model detects:

* Piano
* Acoustic Guitar
* Electric Guitar
* Violin

---

## System Workflow

1. User uploads an audio file (.wav or .mp3)
2. Audio is converted into a mel-spectrogram
3. CNN model analyzes the spectrogram
4. Instrument prediction is generated
5. Results are displayed with:

   * Detected instrument
   * Confidence scores
   * Intensity bars
   * Audio waveform
6. User can download:

   * JSON report
   * PDF report

---

## Model Details

* Dataset: IRMAS (Instrument Recognition in Musical Audio Signals)
* Input: Mel-spectrogram images (224×224)
* Model: Convolutional Neural Network (CNN)
* Validation Accuracy: ~72%
* Framework: TensorFlow / Keras

---

## Tech Stack

* Python
* TensorFlow / Keras
* Librosa (audio processing)
* Streamlit (web interface)
* Matplotlib (visualization)
* ReportLab (PDF generation)

---

## Project Structure

```
InstruNet-AI/
│
├── app.py
├── requirements.txt
├── README.md
└── model/
    └── instrunet_cnn.keras (stored in Google Drive)
```

---

## Installation and Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/InstruNet-AI.git
cd InstruNet-AI
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
streamlit run app.py
```

---

## Usage

1. Launch the Streamlit app.
2. Upload a `.wav` or `.mp3` audio file.
3. View:

   * Detected instrument
   * Confidence chart
   * Intensity visualization
   * Audio waveform
4. Download JSON or PDF report.

---

## Example Output

Detected Instrument:

```
Piano (0.87 confidence)
```

Instrument Intensity:

```
Piano: |||||||||||||||
Guitar:
Electric Guitar:
Violin: ||
```

---

## Future Improvements

* Multi-instrument detection in a single track
* Timeline-based instrument visualization
* Higher accuracy using larger datasets
* Real-time microphone input
* Mobile or web deployment

---

## Author

**Vignesh B**
AI/ML Project – InstruNet AI

---

## License

This project is for educational and demonstration purposes.



link = https://cnn-based-music-instrument-recognition-system-vignesh-221.streamlit.app/
