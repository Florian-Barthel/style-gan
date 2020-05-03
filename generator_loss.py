import tensorflow as tf
import numpy as np


def G_wgan(D, G, minibatch_size, resolution, lod): # pylint: disable=unused-argument
    latents = tf.random.normal([minibatch_size, resolution, 1])
    lods = np.full((minibatch_size, 1), lod)
    fake_images_out = G([latents, lods], training=True)
    fake_scores_out = D([fake_images_out, lods], training=True)
    loss = -fake_scores_out
    return loss

