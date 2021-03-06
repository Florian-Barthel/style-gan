import tensorflow as tf
import numpy as np


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


def g_logistic_nonsaturating(generator, discriminator, latents, lod):
    fake_images_out = generator([latents, lod, np.float32(0)], trainable=True)
    fake_scores = discriminator([fake_images_out, lod], trainable=True)
    loss = tf.nn.softplus(-fake_scores)
    loss = tf.reduce_mean(loss)
    return loss


def d_logistic_simplegp(generator, discriminator, lod, images, latents, r1_gamma=10.0):
    fake_images_out = generator([latents, lod, np.float32(0)], trainable=True)
    fake_scores = discriminator([fake_images_out, lod], trainable=True)
    with tf.GradientTape() as disc_tape:
        disc_tape.watch(images)
        real_scores = discriminator([images, lod], trainable=True)
        real_loss = tf.reduce_sum(real_scores)

    real_grads = disc_tape.gradient(real_loss, images)
    r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])

    loss = tf.nn.softplus(fake_scores)
    loss += tf.nn.softplus(-real_scores)
    loss += r1_penalty * (r1_gamma * 0.5)
    return tf.reduce_mean(loss)


def d_logistic(generator, discriminator, lod, images, latents):
    fake_images_out = generator([latents, lod, np.float32(0)], trainable=True)
    real_scores = discriminator([images, lod], trainable=True)
    fake_scores = discriminator([fake_images_out, lod], trainable=True)
    loss = tf.nn.softplus(fake_scores)
    loss += tf.nn.softplus(-real_scores)
    return tf.reduce_mean(loss)