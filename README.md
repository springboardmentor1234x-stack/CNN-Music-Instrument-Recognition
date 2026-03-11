# CNN Music Instrument Recognition

This project detects musical instruments using CNN and audio signal processing.

## Dataset
IRMAS (Instrument Recognition in Musical Audio Signals)

Instrument classes:
- cel (cello)
- cla (clarinet)
- flu (flute)
- gac (acoustic guitar)
- gel (electric guitar)
- org (organ)
- pia (piano)
- sax (saxophone)
- tru (trumpet)
- vio (violin)
- voi (voice)

## Preprocessing Workflow

Audio (.wav)
↓
Short-Time Fourier Transform (STFT)
↓
Mel Spectrogram
↓
Log Scaling
↓
Saved as Image

## Current Progress

-> Dataset downloaded  
-> IRMAS folder structure verified  
-> Spectrogram generation completed  

## Dataset Preparation

After generating Mel spectrogram images, the dataset is split into:

- Train (70%)
- Validation (15%)
- Test (15%)

The script `split_dataset.py` automatically organizes the dataset into the required structure for CNN training.
