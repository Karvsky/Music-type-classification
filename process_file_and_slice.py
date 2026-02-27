import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE
target_length = 66150

commands = tf.constant(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])

def process_file_and_slice(file_path):
    label_string = tf.strings.split(file_path, os.path.sep)[-2]
    label_id = tf.argmax(label_string == commands)

    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    waveform = tf.squeeze(audio, axis=-1)

    chunks = tf.signal.frame(waveform, frame_length=target_length, frame_step=target_length, pad_end=True)
    num_chunks = tf.shape(chunks)[0]
    
    labels = tf.repeat(label_id, num_chunks)
    
    return chunks, labels