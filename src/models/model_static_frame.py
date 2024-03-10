import tensorflow as tf

from keras.src.layers import Rescaling


def get_model_static_frame(input_shape, num_classes, seed):
    tf.random.set_seed(seed)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Rescaling(1. / 255),
        tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=num_classes, activation="softmax")
    ])

    return model




