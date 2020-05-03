import tensorflow as tf
import numpy as np

def D_wgan(D, G, minibatch_size, reals, resolution, lod, wgan_epsilon=0.001):
    latents = tf.random.normal([minibatch_size, resolution, 1])
    lods = np.full((minibatch_size, 1), lod)
    fake_images_out = G([latents, lods], training=True)
    real_scores_out = D([reals, lods], training=True)
    fake_scores_out = D([fake_images_out, lods], training=True)
    loss = fake_scores_out - real_scores_out

    epsilon_penalty = tf.square(real_scores_out)
    loss += epsilon_penalty * wgan_epsilon
    return loss
    # return tf.reduce_sum(loss)
