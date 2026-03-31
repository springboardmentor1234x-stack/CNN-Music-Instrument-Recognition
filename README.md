# CNN-Music-Instrument-Recognition

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)

## 📝 Description

CNN-Music-Instrument-Recognition is a deep learning project designed to automate the identification of musical instruments within audio recordings. Developed using Python, this project leverages Convolutional Neural Networks (CNNs) to analyze audio data—often by converting sound into visual representations like Mel-spectrograms—to accurately classify various instrument types. This application bridges the gap between digital signal processing and computer vision, offering a powerful tool for music information retrieval (MIR) and automated metadata generation for audio libraries.

## 🛠️ Tech Stack

- 🐍 Python


## 📦 Key Dependencies

```
librosa: 0.10.2
soundfile: 0.12.1
pydub: 0.25.1
tensorflow: 2.15.0
torch: latest
torchvision: latest
numpy: 1.26.4
matplotlib: 3.8.3
seaborn: 0.13.2
scikit-learn: 1.4.1
Pillow: 10.2.0
plotly: 5.20.0
streamlit: 1.32.2
fpdf2: 2.7.9
tqdm: 4.66.2
```

## 📁 Project Structure

```
.
├── app.py
├── dashboard.py
├── data
│   ├── IRMAS-TrainingData
│   │   └── README.txt
│   └── spectrograms
│       └── dataset_stats.json
├── gpu_check.py
├── models
│   ├── evaluation_results.json
│   ├── instrument_classifier_full.pkl
│   └── label_classes.json
├── packages.txt
├── phase1_preprocessing.py
├── phase2_cnn_model.py
├── phase3_evaluation.py
├── requirements.txt
├── save_model_pkl.py
├── streamlit_app.py
└── templates
    └── index.html
```

## 🛠️ Development Setup

### Python Setup
1. Install Python (v3.8+ recommended)
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`


## 👥 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/springboardmentor1234x-stack/CNN-Music-Instrument-Recognition.git`
3. **Create** a new branch: `git checkout -b feature/your-feature`
4. **Commit** your changes: `git commit -am 'Add some feature'`
5. **Push** to your branch: `git push origin feature/your-feature`
6. **Open** a pull request

Please ensure your code follows the project's style guidelines and includes tests where applicable.

---
