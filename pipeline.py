# pipeline.py
import librosa
import numpy as np
import tensorflow as tf
import os
from datetime import datetime


def preprocess_audio(y, sr=22050):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_min, mel_max = mel_spec_db.min(), mel_spec_db.max()
    if mel_max - mel_min > 1e-6:
        mel_spec_db = (mel_spec_db - mel_min) / (mel_max - mel_min)
    else:
        mel_spec_db = np.zeros_like(mel_spec_db)

    def norm(x):
        xmin, xmax = x.min(), x.max()
        return (x - xmin) / (xmax - xmin + 1e-6)

    delta  = librosa.feature.delta(mel_spec_db)
    delta2 = librosa.feature.delta(mel_spec_db, order=2)

    combined = np.stack(
        [norm(mel_spec_db), norm(delta), norm(delta2)],
        axis=-1
    )
    combined = tf.image.resize(combined, (128, 128)).numpy()
    combined = combined.astype(np.float32)
    return np.expand_dims(combined, axis=0)


def run_pipeline(audio_path, model, class_names,
                 segment_duration=3.0,
                 hop_duration=1.5,
                 threshold=0.15):

    # STEP 1: LOAD AUDIO
    y, sr    = librosa.load(audio_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    # STEP 2: SEGMENTATION
    segment_len = int(segment_duration * sr)
    hop_len     = int(hop_duration * sr)
    segments    = []
    timestamps  = []

    for start in range(0, len(y) - segment_len + 1, hop_len):
        segments.append(y[start: start + segment_len])
        timestamps.append({
            "start": round(start / sr, 2),
            "end"  : round((start + segment_len) / sr, 2)
        })

    if not segments:
        segments.append(y)
        timestamps.append({
            "start": 0.0,
            "end"  : round(duration, 2)
        })

    # STEP 3: FEATURE EXTRACTION + CNN PREDICTION
    all_segment_probs = []
    for segment in segments:
        inp   = preprocess_audio(segment, sr)
        probs = model.predict(inp, verbose=0)[0]
        all_segment_probs.append(probs.tolist())

    # STEP 4: AGGREGATION
    mean_probs   = np.mean(all_segment_probs, axis=0)
    detected_idx = np.where(mean_probs > threshold)[0]

    if len(detected_idx) == 0:
        detected_idx = [np.argmax(mean_probs)]

    detected = [class_names[i] for i in detected_idx]

    # STEP 5: BUILD RESULT OBJECT
    # mentor requirement:
    # metadata, model_config, predictions, confidence, timelines
    report = {

        # METADATA
        "metadata": {
            "audio_name"        : os.path.basename(audio_path),
            "audio_duration_sec": round(duration, 2),
            "report_generated"  : datetime.now().strftime(
                                   "%Y-%m-%d %H:%M:%S")
        },

        # MODEL CONFIGURATION
        "model_config": {
            "threshold"           : threshold,
            "segment_duration_sec": segment_duration,
            "hop_duration_sec"    : hop_duration,
            "total_segments"      : len(segments),
            "aggregation_method"  : "mean"
        },

        # PREDICTIONS
        "predictions": {
            "detected_instruments": detected,
            "instrument_wise": {
                name: {
                    "status"          : "Present" if name in detected
                                        else "Not Present",
                    "confidence_score": round(float(mean_probs[i]), 4),
                    "confidence_pct"  : f"{mean_probs[i]*100:.1f}%"
                }
                for i, name in enumerate(class_names)
            }
        },

        # CONFIDENCE SCORES
        "confidence": {
            name: round(float(mean_probs[i]), 4)
            for i, name in enumerate(class_names)
        },

        # TIMELINES
        "timelines": [
            {
                "segment"       : idx,
                "time_start"    : timestamps[idx]["start"],
                "time_end"      : timestamps[idx]["end"],
                "top_instrument": class_names[np.argmax(seg_probs)],
                "probabilities" : {
                    name: round(float(prob), 4)
                    for name, prob in zip(class_names, seg_probs)
                }
            }
            for idx, seg_probs in enumerate(all_segment_probs)
        ]
    }

    print(f"File     : {os.path.basename(audio_path)}")
    print(f"Duration : {round(duration, 2)}s")
    print(f"Segments : {len(segments)}")
    print(f"Detected : {detected}")

    return report, mean_probs, all_segment_probs
