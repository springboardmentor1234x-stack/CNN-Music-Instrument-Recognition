import tensorflow as tf
from tensorflow.keras import layers, regularizers

L2 = 0.001

model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),

    # Block 1
    layers.Conv2D(32, (3,3), padding='same',
                  kernel_regularizer=regularizers.l2(L2)),
    layers.LeakyReLU(0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3,3), padding='same',
                  kernel_regularizer=regularizers.l2(L2)),
    layers.LeakyReLU(0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 3
    layers.Conv2D(128, (3,3), padding='same',
                  kernel_regularizer=regularizers.l2(L2)),
    layers.LeakyReLU(0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),


    layers.Conv2D(256, (3,3), padding='same',
                  kernel_regularizer=regularizers.l2(L2)),
    layers.LeakyReLU(0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3,3), padding='same',
                  kernel_regularizer=regularizers.l2(L2)),
    layers.LeakyReLU(0.1),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, kernel_regularizer=regularizers.l2(L2)),
    layers.LeakyReLU(0.1),
    layers.Dropout(0.5),

    layers.Dense(11, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=6,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

history_r1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

r1_val = max(history_r1.history['val_accuracy'])
r1_trn = max(history_r1.history['accuracy'])
print(f"Train Accuracy : {r1_trn:.4f}")
print(f"Val   Accuracy : {r1_val:.4f}")


model.save("/content/drive/MyDrive/instrunet_model_best.h5")
print("Model saved!")






