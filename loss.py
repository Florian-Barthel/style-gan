import tensorflow as tf
import config
import numpy as np


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


def d_logistic(real_scores_out, fake_scores_out):
    loss = tf.nn.softplus(fake_scores_out)
    loss += tf.nn.softplus(-real_scores_out)
    return tf.reduce_mean(loss)


def g_logistic_nonsaturating(generator, discriminator, latents, lod):
    fake_images_out = generator([latents, lod])
    fake_scores = discriminator([fake_images_out, lod])
    loss = tf.nn.softplus(-fake_scores)
    loss = tf.reduce_mean(loss)
    return loss


def d_logistic_simplegp(generator, discriminator, lod, images, latents, r1_gamma=10.0, r2_gamma=0.0):
    fake_images_out = generator([latents, lod])
    real_scores = discriminator([images, lod])
    fake_scores = discriminator([fake_images_out, lod])
    loss = tf.nn.softplus(fake_scores)
    loss += tf.nn.softplus(-real_scores)
    loss = tf.reduce_mean(loss)
    return loss
