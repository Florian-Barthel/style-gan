import tensorflow as tf


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


def d_logistic(real_scores_out, fake_scores_out):
    loss = tf.nn.softplus(fake_scores_out)
    loss += tf.nn.softplus(-real_scores_out)
    return tf.reduce_mean(loss)


def g_logistic(fake_scores_out):
    loss = tf.nn.softplus(-fake_scores_out)
    return tf.reduce_mean(loss)
