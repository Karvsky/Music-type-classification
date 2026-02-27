import tensorflow as tf

def get_spectrogram(waveform, label):

    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    spectrogram = spectrogram[..., tf.newaxis]
    
    return spectrogram, label