import tensorflow as tf


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)