import tensorflow as tf
from data_preparation import ready_data
from neural_network import neural_network
import numpy as np

neural_network = neural_network()

train, val, test = ready_data()

neural_network.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = neural_network.fit(
    train,
    validation_data=val,
    epochs=20,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=3),
)

loss, accuracy = neural_network.evaluate(test)
print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")

commands = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
y_true = []
y_pred = []

for x, y in test:
    predictions = neural_network.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nAccuracy per genre:")
for i, genre in enumerate(commands):
    indices = np.where(y_true == i)[0]
    if len(indices) > 0:
        genre_acc = np.sum(y_pred[indices] == y_true[indices]) / len(indices)
        print(f"{genre:<12}: {genre_acc*100:.1f}%")