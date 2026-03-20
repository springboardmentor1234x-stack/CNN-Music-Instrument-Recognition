
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

#The above Cnn-model showed training accuracy of "86%" and validation accuracy of "79.6%" , hyperparameters tunning has been done step by step ,initially the base model had about "52%" of validation accuracy and was overfitting.
#In the train model ,more data augmentations techniques were used to improve  generalization and batch-size as been given "32" (32 was selected as it provided optimal balance between training stability and cnn performance),
#In the cnn_model 'batch-normalization' , 'leakyReLu' , 'maxpooling' , 'conv2D' is added and i am using 'softmax' here for the multi-class classification and 'adam' as the optimizer and early stopping was implementaed to halt training to halt when validation performance stopped improving.
#Dropout layer played an important role in increasing the accuracy and reducing the overfitting. "0.04" was used earlier but using '0.03" (overfitting was also reduced) gave a rise in the validation accuracy.
#Changing the Optimizer 'adam' leaning rate also played an important role , also the learning rate was tested from 0.0005 and 0.001 was considered the optimal.
#Initially 10 epoch was used for testing ,few changes in the data augmentation and adding layers showed an accuracy of "58" and "54" which was decrase in the training accuracy but increase in validation and was not overfitting, later dropout layer and one more layer and leakyReLu maxpooling and early stopping.
#Overall the accuracy was increased from 52 to 79(the validation accuracy) and the model does not show anymore overfitting.   
    


    
