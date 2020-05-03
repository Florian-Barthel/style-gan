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
