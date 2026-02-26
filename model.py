import tensorflow as tf
import numpy as np


path = ".\Data\genres_original"

commands = np.array(tf.io.gfile.listdir(str(path)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(path) + '/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(path/commands[0]))))
print('Example file tensor:', filenames[0])