
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    tf.keras.Input(shape=(128,128,3)),
    layers.Conv2D(32,(3,3), padding='same'),
    layers.LeakyReLU(negative_slope=0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3), padding='same'),
    layers.LeakyReLU(negative_slope=0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3), padding='same'),
    layers.LeakyReLU(negative_slope=0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(256,(3,3), padding='same'),
    layers.LeakyReLU(negative_slope=0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(512,(3,3), padding='same'),
    layers.LeakyReLU(negative_slope=0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.GlobalAveragePooling2D(),

    layers.Dense(512),
    layers.LeakyReLU(negative_slope=0.1),
    layers.Dropout(0.3),

    layers.Dense(11,activation='softmax')
])
model.compile(
   optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1

)

history_adam = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[early_stop,reduce_lr],

)

print("Best Training Accuracy:",
      max(history_adam.history['accuracy']))

print("Best Validation Accuracy:",
      max(history_adam.history['val_accuracy']))

    