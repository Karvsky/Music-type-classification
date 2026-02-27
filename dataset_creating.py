import tensorflow as tf
from process_file_and_slice import process_file_and_slice

AUTOTUNE = tf.data.AUTOTUNE
target_length = 66150

def create_dataset(files):

    ds = tf.data.Dataset.from_tensor_slices(files)

    ds = ds.map(process_file_and_slice, num_parallel_calls=AUTOTUNE)

    ds = ds.unbatch()
    return ds
