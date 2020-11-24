import tensorflow as tf

def downsample(filters, size, stride):
    conv = tf.keras.layers.Conv2D(filters, size, activation='relu', strides=stride, padding='same')
    return conv


def upsample(filters, size, stride):
    deconv = tf.keras.layers.Conv2DTranspose(filters, size, activation='relu', strides=stride, padding='same')
    return deconv