import tensorflow as tf
from functools import partial

from keras.src.layers import Rescaling

static_frame_model = None


def get_or_create_model_static_frame(input_shape, num_classes, seed):

    global static_frame_model

    if static_frame_model is not None:
        return static_frame_model

    DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                            activation="relu", kernel_initializer="he_normal")

    tf.random.set_seed(seed)
    static_frame_model = tf.keras.Sequential([
        Rescaling(1. / 255, input_shape=input_shape),
        DefaultConv2D(filters=64, kernel_size=7),
        tf.keras.layers.MaxPool2D(),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        tf.keras.layers.MaxPool2D(),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation="relu",
                              kernel_initializer="he_normal"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=64, activation="relu",
                              kernel_initializer="he_normal"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=num_classes, activation="softmax")
    ])

    return static_frame_model




