
import json
import numpy as np
from tensorflow.keras.models import load_model

instruments = ["piano","guitar","drums","violin","bass"]

def predict_from_spectrogram(spec):

    model = load_model("models/instrument_model.h5")

    spec = spec.reshape(1,128,128,1)

    preds = model.predict(spec)[0]

    result = {}

    for i,v in enumerate(preds):
        if v > 0.5:
            result[instruments[i]] = float(v)

    with open("outputs/prediction.json","w") as f:
        json.dump(result,f,indent=4)

    return result
