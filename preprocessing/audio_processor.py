import librosa
import numpy as np

def load_and_preprocess_audio(file_path, target_sr=22050):
    """
    Loads an audio file, converts it to mono, resamples to target_sr, and normalizes it.
    
    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sampling rate. Default is 22050 Hz.
        
    Returns:
        audio (np.ndarray): Preprocessed audio signal.
        sr (int): Sampling rate used.
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Normalize the audio signal to range [-1.0, 1.0]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        return audio, sr
    except Exception as e:
        print(f"Error processing audio file {file_path}: {e}")
        return None, None

def split_audio_into_segments(audio, sr, segment_duration=3.0):
    """
    Splits the audio signal into smaller time segments.
    
    Args:
        audio (np.ndarray): The audio signal.
        sr (int): Sampling rate.
        segment_duration (float): Duration of each segment in seconds.
        
    Returns:
        list of np.ndarray: List containing audio segments.
    """
    segment_length = int(sr * segment_duration)
    total_length = len(audio)
    
    segments = []
    for start in range(0, total_length, segment_length):
        end = start + segment_length
        segment = audio[start:end]
        
        # Pad the last segment if it's shorter than the required length
        if len(segment) < segment_length:
            padding = segment_length - len(segment)
            segment = np.pad(segment, (0, padding), mode='constant')
            
        segments.append(segment)
        
    return segments

def generate_mel_spectrogram(segment, sr, n_mels=128, hop_length=512):
    """
    Generates a normalized Mel-Spectrogram for a given audio segment.
    
    Args:
        segment (np.ndarray): Audio segment.
        sr (int): Sampling rate.
        n_mels (int): Number of Mel bands.
        hop_length (int): Number of audio samples between adjacent STFT columns.
        
    Returns:
        np.ndarray: Normalized log-mel spectrogram [Shape: (n_mels, time_frames, 1)]
    """
    # Compute the Mel-scaled power spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    
    # Convert power spectrogram to decibel (log scale)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Normalize to [0, 1] range using min-max scaling
    # librosa.power_to_db produces values in roughly [-80, 0] range.
    # We use fixed bounds to ensure consistent scaling across all samples,
    # which preserves the relative spectral differences between instruments.
    db_min = -80.0
    db_max = 0.0
    log_spectrogram = np.clip(log_spectrogram, db_min, db_max)
    log_spectrogram = (log_spectrogram - db_min) / (db_max - db_min)
    
    # Add channel dimension for CNN compatibility (Grayscale image)
    log_spectrogram = np.expand_dims(log_spectrogram, axis=-1)
    
    return log_spectrogram

def process_pipeline(file_path, segment_duration=3.0):
    """
    End-to-end preprocessing pipeline for a single audio file.
    
    Args:
        file_path (str): Path to audio file.
        segment_duration (float): Duration for splitting.
        
    Returns:
        np.ndarray: Array of spectrograms for all segments.
    """
    audio, sr = load_and_preprocess_audio(file_path)
    if audio is None:
        return None
        
    segments = split_audio_into_segments(audio, sr, segment_duration)
    
    spectrograms = []
    for seg in segments:
        spec = generate_mel_spectrogram(seg, sr)
        spectrograms.append(spec)
        
    # Convert list of spectrograms to a single numpy array batch
    return np.array(spectrograms)
