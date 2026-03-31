import numpy as np
import json

instruments = ["piano","guitar","drums","violin","bass"]

def aggregate(preds):
    return np.mean(preds, axis=0)

def apply_threshold(avg, thresh=0.5):
    result = {}
    for i,v in enumerate(avg):
        if v > thresh:
            result[instruments[i]] = float(v)
    return result

def save_json(result):
    with open("outputs/result.json","w") as f:
        json.dump(result,f,indent=4)