import tensorflow as tf


def D_wgan(real_scores_out, fake_scores_out, wgan_epsilon=0.001):
    loss = fake_scores_out - real_scores_out
    epsilon_penalty = tf.square(real_scores_out)
    loss += epsilon_penalty * wgan_epsilon
    return loss


def D_logistic_simplegp(real_scores_out, fake_scores_out):
    loss = tf.nn.softplus(fake_scores_out)
    loss += tf.nn.softplus(-real_scores_out)
    return loss


def D_logistic_simplegp_only_real(real_scores_out):
    loss = tf.nn.softplus(-real_scores_out)
    return loss


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def cross_entropy_loss(real_scores_out, fake_scores_out):
    real_loss = cross_entropy(tf.ones_like(real_scores_out), real_scores_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_scores_out), fake_scores_out)
    total_loss = real_loss + fake_loss
    return total_loss