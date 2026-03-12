# os module is used to interact with folders and file paths
import os

# librosa is a library used for audio processing
import librosa

# soundfile is used to save audio files after processing
import soundfile as sf


# function that will preprocess all audio files
def preprocess_audio(input_path, output_path):

    # create the processed folder if it does not already exist
    os.makedirs(output_path, exist_ok=True)

    # loop through each instrument folder (cel, cla, flu, etc.)
    for instrument in os.listdir(input_path):

        # create the full path of the instrument folder
        instrument_path = os.path.join(input_path, instrument)

        # check if it is actually a folder
        if not os.path.isdir(instrument_path):
            continue   # skip if it is not a directory

        # create corresponding folder in processed directory
        save_folder = os.path.join(output_path, instrument)

        # create the folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)

        # loop through each file inside the instrument folder
        for file in os.listdir(instrument_path):

            # process only .wav audio files
            if file.endswith(".wav"):

                # create full path of the audio file
                file_path = os.path.join(instrument_path, file)

                # load the audio file
                # y = audio signal (array of sound values)
                # sr = sampling rate (samples per second)
                # mono=True converts stereo audio into mono
                y, sr = librosa.load(file_path, sr=22050, mono=True)

                # normalize the audio signal
                # this keeps volume levels consistent
                y = librosa.util.normalize(y)

                # create output file path
                save_path = os.path.join(save_folder, file)

                # save the processed audio file
                sf.write(save_path, y, sr)


# this block runs only when the file is executed directly
if __name__ == "__main__":

    # location of original IRMAS dataset
    input_path = "data/raw/IRMAS-TrainingData"

    # location where processed audio will be saved
    output_path = "data/processed"

    # call the preprocessing function
    preprocess_audio(input_path, output_path)

    # print message when processing is finished
    print("Audio preprocessing completed.")