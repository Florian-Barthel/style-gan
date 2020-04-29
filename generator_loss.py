import tensorflow as tf


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)