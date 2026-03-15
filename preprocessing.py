# os is used for working with folders and file paths
import os

# librosa is used for audio processing
import librosa

# soundfile is used to save processed audio
import soundfile as sf


# function to preprocess audio files
def preprocess_audio(input_path, output_path):

    # create processed folder if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # loop through each instrument folder
    for instrument in os.listdir(input_path):

        instrument_path = os.path.join(input_path, instrument)

        # skip if it is not a folder
        if not os.path.isdir(instrument_path):
            continue

        # create corresponding processed folder
        save_folder = os.path.join(output_path, instrument)
        os.makedirs(save_folder, exist_ok=True)

        # loop through each audio file
        for file in os.listdir(instrument_path):

            # process only wav files
            if file.endswith(".wav"):

                # full path of audio file
                file_path = os.path.join(instrument_path, file)

                # load audio
                # y = audio signal
                # sr = sampling rate
                y, sr = librosa.load(file_path, sr=22050, mono=True)

                # -------- TRIM SILENCE --------
                # removes silent parts from beginning and end
                y_trimmed, _ = librosa.effects.trim(y)

                # normalize audio volume
                y_normalized = librosa.util.normalize(y_trimmed)

                # save processed audio
                save_path = os.path.join(save_folder, file)
                sf.write(save_path, y_normalized, sr)


# run the script
if __name__ == "__main__":

    # path of raw dataset
    input_path = "data/raw/IRMAS-TrainingData"

    # path where processed audio will be saved
    output_path = "data/processed"

    preprocess_audio(input_path, output_path)

    print("Audio preprocessing with silence trimming completed.")
