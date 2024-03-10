import tensorflow as tf
from tensorflow.keras import layers, models

from keras.src.layers import Rescaling


def get_model_static_frame(input_shape, num_classes, seed):
    tf.random.set_seed(seed)

    model = models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Rescaling(1. / 255),
        # Convolutional layers
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        # Flatten layer
        layers.Flatten(),
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model




