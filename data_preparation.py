import tensorflow as tf
import numpy as np
import os

AUTOTUNE = tf.data.AUTOTUNE
target_length = 66150

def process_file_and_slice(file_path):

    label = tf.strings.split(file_path, os.path.sep)[-2]

    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    waveform = tf.squeeze(audio, axis=-1)

    chunks = tf.signal.frame(waveform, frame_length=target_length, frame_step=target_length, pad_end=True)

    num_chunks = tf.shape(chunks)[0]
    labels = tf.repeat(label, num_chunks)
    
    return chunks, labels

def create_dataset(files):

    ds = tf.data.Dataset.from_tensor_slices(files)

    ds = ds.map(process_file_and_slice, num_parallel_calls=AUTOTUNE)

    ds = ds.unbatch()
    return ds

def get_spectrogram(waveform, label):

    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    spectrogram = spectrogram[..., tf.newaxis]
    
    return spectrogram, label

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