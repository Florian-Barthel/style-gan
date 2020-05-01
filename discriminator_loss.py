import tensorflow as tf


def D_wgan(D, G, minibatch_size, reals, resolution, wgan_epsilon=0.001):
    latents = tf.random.normal([minibatch_size, resolution, 1])
    fake_images_out = G(latents, training=True)
    real_scores_out = D(reals, training=True)
    fake_scores_out = D(fake_images_out, training=True)
    loss = fake_scores_out - real_scores_out

    epsilon_penalty = tf.square(real_scores_out)
    loss += epsilon_penalty * wgan_epsilon
    return tf.reduce_sum(loss)
