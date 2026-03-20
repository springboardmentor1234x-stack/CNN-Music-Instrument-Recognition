import json
import numpy as np
from tensorflow.keras.models import load_model

instruments = ["piano","guitar","drums","violin","bass"]


def smooth_predictions(preds):
    smoothed = []
    for i in range(1, len(preds)-1):
        if preds[i-1] == preds[i+1]:
            smoothed.append(preds[i-1])
        else:
            smoothed.append(preds[i])
    return smoothed


def predict_from_spectrogram(spec):

    model = load_model("models/instrument_model_adam.h5")

    spec = spec.reshape(1,128,128,1)

    preds = model.predict(spec)[0]

    print("Predicted probabilities:", preds)

    result = {}

    for i,v in enumerate(preds):
        if v > 0.5:
            result[instruments[i]] = float(v)

    with open("outputs/prediction.json","w") as f:
        json.dump(result,f,indent=4)

    return result