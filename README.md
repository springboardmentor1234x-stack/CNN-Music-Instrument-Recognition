
# InstruNet AI 🎵

InstruNet AI is a deep learning project that detects musical instruments in an audio track using CNNs.

## Features
- Audio preprocessing
- Mel spectrogram generation
- CNN-based instrument classification
- Multi-instrument prediction
- Visualization of detected instruments
- JSON report generation

## Project Structure
```
data/
  raw_audio/
  spectrograms/
models/
outputs/
src/
  preprocess.py
  spectrogram.py
  train_model.py
  predict.py
  visualize.py
requirements.txt
run_pipeline.py
```

## Setup

1. Install dependencies

```
pip install -r requirements.txt
```

2. Add audio files to:

```
data/raw_audio/
```

3. Train model

```
python src/train_model.py
```

4. Run prediction

```
python src/predict.py
```

Outputs will be saved in the `outputs/` folder.
