import tensorflow as tf
from data_preparation import ready_data

def neural_network():

    train, val, test = ready_data()

    input_shape = (515, 129, 1) 
    num_labels = 10

    normalization_layer = tf.keras.layers.Normalization()

    normalization_layer.adapt(data=train.map(map_func=lambda spec, label: spec))

    input_shape = (515, 129, 1)
    num_labels = 10

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Resizing(128, 64),
        normalization_layer,
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_labels),
    ])

    return model

