import tensorflow as tf


def G_wgan(D, G, minibatch_size, resolution): # pylint: disable=unused-argument
    latents = tf.random.normal([minibatch_size, resolution, 1])
    fake_images_out = G(latents, training=True)
    fake_scores_out = D(fake_images_out, training=True)
    loss = -fake_scores_out
    return tf.reduce_sum(loss)
