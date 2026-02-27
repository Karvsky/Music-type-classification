import tensorflow as tf
from data_preparation import ready_data
from neural_network import neural_network

neural_network = neural_network()

train, val, test = ready_data()

neural_network.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = neural_network.fit(
    train,
    validation_data=val,
    epochs=20,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=3),
)