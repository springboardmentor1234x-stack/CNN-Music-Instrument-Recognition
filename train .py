import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

dataset_path = "/content/drive/MyDrive/spectrogram_datasetss"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    width_shift_range=0.1,    
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    subset='training',
    shuffle=True,
    seed=42
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    subset='validation',
    shuffle=False,
    seed=42
)

# CLASS WEIGHTS
class_labels      = train_data.classes
classes           = np.unique(class_labels)
weights           = compute_class_weight('balanced', classes=classes, y=class_labels)
class_weight_dict = dict(zip(classes, weights))

print("Class indices :", train_data.class_indices)
print("Class weights :", {k: round(v, 3) for k, v in class_weight_dict.items()})
