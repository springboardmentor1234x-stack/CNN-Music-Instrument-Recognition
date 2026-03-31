# InstruNet AI 🎵

InstruNet AI is a deep learning-based system designed to automatically detect musical instruments present in an audio track using Convolutional Neural Networks (CNNs). The system converts audio into mel-spectrogram images and performs multi-label classification to identify instruments such as piano, guitar, drums, violin, and bass.

---

## 🚀 Features

* Audio preprocessing (normalization, silence removal)
* Mel spectrogram generation with log scaling
* CNN-based multi-label classification
* Multi-instrument detection
* Visualization of spectrograms
* JSON-based prediction output
* Segment-based audio processing
* Prediction averaging and smoothing

---

## 📁 Project Structure

```
InstruNet/
├── app.py
├── requirements.txt
├── data/
│ └── sample.wav
├── outputs/
│ ├── result.json
│ └── report.pdf
└── src/
├── preprocess.py
├── spectrogram.py
├── pipeline.py
├── predict.py
└── report.py
```

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Add audio files

Place your audio files inside:

```
data/raw_audio/
```

---

## 🧠 Model Training

Train the CNN model with different optimizers:

```
python src/train_model.py
```

This will:

* Train models using SGD, RMSprop, and Adam
* Save trained models in the `models/` folder
* Generate training vs validation loss plots

---

## 🔮 Prediction Pipeline

Run the Streamlit app:


streamlit run app.py


This performs:

- Audio preprocessing  
- Segmentation of audio  
- Spectrogram generation  
- Instrument prediction  
- Averaging and smoothing  

Outputs are saved in:


outputs/

---
## 🖥️ Streamlit Dashboard

- Upload audio files (.wav)
- View audio playback
- Run full prediction pipeline
- View detected instruments
- Generate JSON output
- Generate PDF report

## 📊 Advanced Features

* Hyperparameter tuning (learning rate, optimizers)
* Comparison of SGD, RMSprop, and Adam
* Training & validation loss tracking
* Early stopping (callbacks)
* Segment-wise prediction
* Averaging and smoothing of predictions
* Confusion matrix generation
* Classification metrics (precision, recall, F1-score)
* Streamlit-based interactive dashboard
* PDF report generation

---

## 📈 Results

* Adam optimizer achieved the best validation performance
* Segment-based prediction improved stability
* Smoothing reduced noise in predictions
Note: Currently the model is trained on synthetic (dummy) data. Performance can be significantly improved using real-world labeled datasets.
---

## 🧠 Conclusion

This project demonstrates an end-to-end deep learning pipeline for music instrument recognition. By combining CNN-based feature extraction with optimization techniques and segment-level analysis, the system achieves improved accuracy and robustness.

---

## 🔮 Future Improvements

* Training on real-world datasets instead of synthetic data
* Data augmentation techniques
* ROC curve analysis for each instrument
* Deployment as a web or mobile application
