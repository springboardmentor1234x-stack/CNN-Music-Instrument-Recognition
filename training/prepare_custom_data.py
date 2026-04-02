import os
import json
import glob

def generate_metadata_from_folders(data_dir="data/real_instruments"):
    """
    Scans internal folders in `data_dir` where each subfolder is an instrument class.
    Generates the required `metadata.json` for training.
    
    Example Structure:
    data/real_instruments/
      ├── guitar/
      │     ├── sound1.wav
      │     └── sound2.wav
      ├── piano/
      │     └── keys.wav
      └── drums/
            └── beat.wav
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory {data_dir}. Please place your instrument audio folders here.")
        return False
        
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(classes) == 0:
        print(f"No class folders found in {data_dir}. Please create folders like 'piano', 'guitar' and put .wav/.mp3 files in them.")
        return False
        
    classes.sort()
    
    metadata = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        audio_files = glob.glob(os.path.join(class_dir, "*.wav")) + glob.glob(os.path.join(class_dir, "*.mp3"))
        
        for filepath in audio_files:
            # Create a one-hot label array
            labels = [0] * len(classes)
            labels[class_idx] = 1
            
            # Use relative path for metadata
            rel_filepath = os.path.relpath(filepath, start=data_dir)
            
            metadata.append({
                "filename": os.path.basename(filepath),
                "filepath": os.path.join("data", "real_instruments", rel_filepath).replace('\\', '/'),
                "class_name": class_name,
                "labels": labels
            })
            
    if len(metadata) == 0:
        print("Folders found, but no .wav or .mp3 files inside them!")
        return False
        
    # Save the custom metadata
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Save the class mapping for inference
    class_mapping = {str(i): name for i, name in enumerate(classes)}
    with open(os.path.join("models", "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=4)
        
    print(f"Success! Found {len(metadata)} audio files across {len(classes)} classes: {', '.join(classes)}")
    return True

if __name__ == "__main__":
    if generate_metadata_from_folders():
        print("\nMetadata generated successfully. You can now run `python training/train.py`")
