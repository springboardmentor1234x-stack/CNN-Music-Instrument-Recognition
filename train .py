
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = "/content/drive/MyDrive/spectrogram_datasets"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=8, #8
    width_shift_range=0.05, 
    height_shift_range=0.05,
    horizontal_flip=False,
   zoom_range=0.1,
    brightness_range=[0.8, 1.2]


)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    subset='validation'
)
