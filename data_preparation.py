import tensorflow as tf
import numpy as np
import os
from process_file_and_slice import process_file_and_slice
from dataset_creating import create_dataset
from spectogram import get_spectrogram

AUTOTUNE = tf.data.AUTOTUNE
target_length = 66150



def ready_data():

      path = r"..\..\Data\genres_original"

      commands = np.array(tf.io.gfile.listdir(str(path)))
      commands = commands[commands != 'README.md']
      commands = commands[commands != '.vscode']

      filenames = tf.io.gfile.glob(str(path) + '/*/*.wav')

      filenames = tf.random.shuffle(filenames)
      num_samples = len(filenames)

      train_files = filenames[:800]
      val_files = filenames[800: 900]
      test_files = filenames[-100:]

      train_ds = create_dataset(train_files)
      val_ds = create_dataset(val_files)
      test_ds = create_dataset(test_files)

      train_ds = train_ds.map(get_spectrogram, num_parallel_calls=AUTOTUNE)
      val_ds = val_ds.map(get_spectrogram, num_parallel_calls=AUTOTUNE)
      test_ds = test_ds.map(get_spectrogram, num_parallel_calls=AUTOTUNE)

      batch_size = 32

      train_ds = train_ds.batch(batch_size).cache().prefetch(AUTOTUNE)
      val_ds = val_ds.batch(batch_size).cache().prefetch(AUTOTUNE)
      test_ds = test_ds.batch(batch_size).cache().prefetch(AUTOTUNE)

      return train_ds, val_ds, test_ds