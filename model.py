import tensorflow as tf
import numpy as np

path = r"..\..\Data\genres_original"

commands = np.array(tf.io.gfile.listdir(str(path)))
commands = commands[commands != 'README.md']
commands = commands[commands != '.vscode']
print('Commands:', commands)

###
filenames = tf.io.gfile.glob(str(path) + '/*/*.wav')
###
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)

###
print('Number of examples per label:',
      len(tf.io.gfile.glob(str(path) + '/' + commands[0] + '/*.wav')))
###


train_files = filenames[:800]
val_files = filenames[800: 900]
test_files = filenames[-100:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))