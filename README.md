**CNN Music Instrument Recognition System**

**1. Overview**

This project implements a complete end-to-end deep learning pipeline for classifying musical instrument sounds using the IRMAS dataset.

The system converts raw audio signals into spectrogram representations and trains a Convolutional Neural Network (CNN) using TensorFlow / Keras.


**2. Key Features**
  * Audio preprocessing and feature extraction (Mel Spectrograms)
  * Custom CNN architecture for classification
  * Training, validation, and evaluation pipeline
  * Organized dataset handling (train/test split)
  * Streamlit-based web app for real-time prediction
  * Public deployment using ngrok/ cloudflare


**3. Dataset**
   Dataset Used: IRMAS (Instrument Recognition in Musical Audio Signals)
   Classes (11 Instruments):
      * Cello (cel)
      * Clarinet (cla)
      * Flute (flu)
      * Acoustic Guitar (gac)
      * Electric Guitar (gel)
      * Organ (org)
      * Piano (pia)
      * Saxophone (sax)
      * Trumpet (tru)
      * Violin (vio)
      * Voice (voi)


**4. Tech stack**
  *  Programming Language: Python
  * Libraries:
      * TensorFlow / Keras
      * Librosa (audio processing)
      * NumPy, Pandas
      * Matplotlib / Seaborn
      * Streamlit


**5. Methodology**
 ** Pipeline Workflow**

  1. Data Collection
      * IRMAS dataset (instrument classification dataset) contains .wav audio files across 11 classes

  2. Data Preprocessing
      * Load audio using librosa
      * Convert audio → Mel Spectrogram
      * Normalize and resize to fixed dimensions (e.g., 128×128)

  3. Dataset Preparation
      * Split into:
        * Training set
        * Testing set
        * Validation set
    

  4. Model Architecture
      A custom CNN with:
        * 4 Convolutional Blocks:
        * Conv2D + BatchNorm + ReLU + MaxPooling + Dropout
        * Global Average Pooling
        * Fully Connected Dense Layers
        * Output Layer (11 classes)

  5. Training
        * Optimizer: Adam
        * Loss Function: categorical_crossentropy
        * Metrics: Accuracy

  6. Evaluation
        * Training and validation accuracy and loss curves
        * Confusion matrix analysis

  7. Deployment
        * Built using Streamlit
        * Users can upload .wav files(.wav, .mp3)
        * Model prediction instrument in real-time

  8. Results
        * Achieved reliable classification performance on IRMAS dataset
        * Model accuracy depends on preprocessing quality and dataset balance
     

**6. Web Application**
      * Upload audio file (.wav / .mp3)
      * Model predicts instrument class
      * Displays prediction confidence
      * Adjustable detection threshold
      * Clean dashboard UI with model status
  
  
**7. Key Learnings**
      * Feature engineering is critical for audio data
      * Spectrogram representation significantly improves model performance
      * Data quality impacts accuracy more than model complexity
      * Proper loss function and activation selection are essential


**8. Future Improvements**
      * Use advanced models (YAMNet, Transformers)
      * Improve dataset balancing and augmentation
      * Add top-k predictions with confidence scores
      * Deploy on cloud platforms (Streamlit Cloud / Hugging Face)
      * Enable real-time audio stream prediction

**9. Author**
S. AISWARYA
Deep Learning Project – Audio Classification

**10. License**
This project was developed as part of the Infosys Springboard Virtual Internship 6.0.

It is intended for educational and non-commercial use only.
All datasets and third-party library rights belong to their respective owners.


