
import os
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def build_cnn(input_shape=(128,128,1), classes=5):

    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(classes,activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_dummy():

    X = np.random.rand(200,128,128,1)
    y = np.random.randint(0,2,(200,5))

    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)

    model = build_cnn()

    model.fit(X_train,y_train,epochs=5,validation_data=(X_val,y_val))

    os.makedirs("models",exist_ok=True)
    model.save("models/instrument_model.h5")

if __name__ == "__main__":
    train_dummy()
