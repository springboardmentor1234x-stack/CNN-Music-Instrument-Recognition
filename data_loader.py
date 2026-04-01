# os module helps us interact with the operating system
# used for reading folders and creating file paths
import os

# pandas is used to create and manage dataset tables
import pandas as pd


# function to load the IRMAS dataset
def load_irmas_dataset(dataset_path):

    # create an empty list where we will store dataset information
    data = []

    # loop through each instrument folder (cel, cla, flu, etc.)
    for instrument in os.listdir(dataset_path):

        # create full path to that instrument folder
        instrument_path = os.path.join(dataset_path, instrument)

        # check if the path is actually a directory (not a file)
        if os.path.isdir(instrument_path):

            # loop through each file inside the instrument folder
            for file in os.listdir(instrument_path):

                # process only .wav audio files
                if file.endswith(".wav"):

                    # create the complete path to the audio file
                    file_path = os.path.join(instrument_path, file)

                    # append the file path and instrument label to the list
                    data.append({
                        "file_path": file_path,   # path of the audio file
                        "instrument": instrument  # instrument label (folder name)
                    })


    # convert the list of dictionaries into a pandas DataFrame
    # DataFrame = structured dataset table
    df = pd.DataFrame(data)

    # return the dataset table
    return df



# this block runs only when the script is executed directly
if __name__ == "__main__":

    # path where the IRMAS training dataset is stored
    dataset_path = "C:/Users/NITIKA KUMARI/instrunet-ai/data/raw/IRMAS-TrainingData"

    # call the function to load dataset
    df = load_irmas_dataset(dataset_path)

    # print confirmation message
    print("Dataset Loaded")

    # print total number of audio files
    print("Total Samples:", len(df))

    # show first 5 rows of dataset
    print(df.head())


    # print instrument distribution
    print("\nInstrument Distribution:")

    # count how many samples exist for each instrument
    print(df["instrument"].value_counts())