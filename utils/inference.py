import os
import numpy as np
import tensorflow as tf
from preprocessing.audio_processor import process_pipeline

def load_inference_model(model_path="models/baseline_cnn.h5"):
    """
    Loads the trained CNN model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

def predict_on_audio(file_path, model, class_mapping, segment_duration=3.0, threshold=0.15):
    """
    Runs time-segment based prediction on an entire audio file.
    
    Args:
        file_path (str): Path to the audio file.
        model (tf.keras.Model): The loaded model.
        class_mapping (dict): Mapping from class index to instrument name.
        segment_duration (float): Duration of each segment.
        threshold (float): Confidence threshold for asserting presence.
        
    Returns:
        dict: A dictionary containing overall predictions and segment-wise details.
    """
    # 1. Preprocess the audio into a batch of spectrograms
    spectrograms = process_pipeline(file_path, segment_duration=segment_duration)
    
    if spectrograms is None or len(spectrograms) == 0:
        return {"error": "Failed to process audio or audio is empty."}
        
    # 2. Predict segment-wise
    # Output shape: (num_segments, num_classes)
    predictions = model.predict(spectrograms, verbose=0)
    
    # 3. Aggregate predictions across all segments
    # For softmax output, use mean confidence across segments
    avg_confidences = np.mean(predictions, axis=0)
    
    num_segments = len(predictions)
    num_classes = len(class_mapping)
    
    # 4. Format segment timeline details
    timeline = []
    for i, seg_preds in enumerate(predictions):
        start_time = i * segment_duration
        end_time = start_time + segment_duration
        
        seg_details = {
            "start": start_time,
            "end": end_time,
            "confidences": {class_mapping[str(idx)]: float(score) 
                           for idx, score in enumerate(seg_preds)},
            "dominant": class_mapping[str(int(np.argmax(seg_preds)))]
        }
        timeline.append(seg_details)
        
    # 5. Format overall results
    # Sort instruments by confidence (highest first)
    detected_instruments = []
    overall_confidences = {}
    
    # Get sorted indices by confidence
    sorted_indices = np.argsort(avg_confidences)[::-1]
    
    for idx in sorted_indices:
        idx_int = int(idx)
        inst_name = class_mapping[str(idx_int)]
        conf = float(avg_confidences[idx_int])
        overall_confidences[inst_name] = conf
        
        if conf >= threshold:
            # Calculate intensity bars (scale to 10 bars)
            bar_count = max(1, int(conf * 10))
            detected_instruments.append({
                "instrument": inst_name,
                "confidence": conf,
                "intensity_bars": "█" * bar_count + "░" * (10 - bar_count)
            })
    
    # If nothing passes threshold, still show top-1 prediction
    if len(detected_instruments) == 0:
        top_idx = int(np.argmax(avg_confidences))
        top_name = class_mapping[str(top_idx)]
        top_conf = float(avg_confidences[top_idx])
        detected_instruments.append({
            "instrument": top_name,
            "confidence": top_conf,
            "intensity_bars": "█" * max(1, int(top_conf * 10)) + "░" * (10 - max(1, int(top_conf * 10)))
        })
            
    return {
        "summary": {
            "detected_instruments": detected_instruments,
            "overall_confidences": overall_confidences
        },
        "timeline": timeline,
        "metadata": {
            "num_segments": num_segments,
            "segment_duration": segment_duration
        }
    }

if __name__ == "__main__":
    pass
