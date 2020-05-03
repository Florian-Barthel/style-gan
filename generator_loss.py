import tensorflow as tf
import numpy as np


def G_wgan(fake_scores_out):
    loss = -fake_scores_out
    return loss

def G_logistic_nonsaturating(fake_scores_out): # pylint: disable=unused-argument
    loss = tf.nn.softplus(-fake_scores_out)  # -log(logistic(fake_scores_out))
    return loss