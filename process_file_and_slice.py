import tensorflow as tf

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