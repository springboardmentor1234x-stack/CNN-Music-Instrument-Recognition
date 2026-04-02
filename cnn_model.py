import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(128, 130, 1), num_classes=7):
    """
    Builds a CNN for spectrogram-based instrument classification.
    
    Uses a straightforward architecture without BatchNormalization to avoid
    train/inference mode discrepancies with small batch sizes.
    
    Args:
        input_shape (tuple): Shape of the input spectrogram (mels, time_steps, channels).
        num_classes (int): Number of output classes (instruments).
        
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Flatten instead of GlobalAveragePooling — preserves spatial info
        layers.Flatten(),

        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
