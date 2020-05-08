import tensorflow as tf

def wasserstein_loss(y_true, y_pred):
    return y_true * tf.reduce_mean(y_pred)