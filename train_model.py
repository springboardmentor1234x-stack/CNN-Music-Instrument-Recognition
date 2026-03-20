import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping
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

    return model


def compile_model(model, optimizer_name):

    if optimizer_name == "sgd":
        opt = SGD(learning_rate=0.01)
    elif optimizer_name == "rmsprop":
        opt = RMSprop(learning_rate=0.001)
    else:
        opt = Adam(learning_rate=0.001)

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(optimizer_name="adam"):

    # Dummy data (replace with real dataset if available)
    X = np.random.rand(200,128,128,1)
    y = np.random.randint(0,2,(200,5))

    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)

    model = build_cnn()
    model = compile_model(model, optimizer_name)

    early_stop = EarlyStopping(patience=3)

    history = model.fit(
        X_train,y_train,
        validation_data=(X_val,y_val),
        epochs=10,
        callbacks=[early_stop]
    )

    os.makedirs("models",exist_ok=True)
    model.save(f"models/instrument_model_{optimizer_name}.h5")

    plot_history(history, optimizer_name)

    return history


def plot_history(history, optimizer_name):

    plt.figure()

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')

    plt.title(f"Loss Curve ({optimizer_name})")
    plt.legend()
    plt.savefig(f"outputs/loss_{optimizer_name}.png")


if __name__ == "__main__":

    for opt in ["sgd","rmsprop","adam"]:
        print(f"Training with {opt}")
        train_model(opt)