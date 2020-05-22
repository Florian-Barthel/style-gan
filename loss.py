import tensorflow as tf


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


def g_logistic_nonsaturating(generator, discriminator, latents, lod):
    fake_images_out = generator([latents, lod])
    fake_scores = discriminator([fake_images_out, lod])
    loss = tf.nn.softplus(-fake_scores)
    loss = tf.reduce_mean(loss)
    return loss


def d_logistic_simplegp(generator, discriminator, lod, images, latents, r1_gamma=10.0):
    fake_images_out = generator([latents, lod])
    fake_scores = discriminator([fake_images_out, lod])
    with tf.GradientTape() as disc_tape:
        disc_tape.watch(images)
        real_scores = discriminator([images, lod])
        real_loss = tf.math.reduce_sum(real_scores)

    real_grads = disc_tape.gradient(real_loss, [images])[0]
    r1_penalty = tf.math.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])

    loss = tf.nn.softplus(fake_scores)
    loss += tf.nn.softplus(-real_scores)
    loss += r1_penalty * (r1_gamma * 0.5)
    return tf.reduce_mean(loss)


def d_logistic(generator, discriminator, lod, images, latents):
    fake_images_out = generator([latents, lod])
    real_scores = discriminator([images, lod])
    fake_scores = discriminator([fake_images_out, lod])
    loss = tf.nn.softplus(fake_scores)
    loss += tf.nn.softplus(-real_scores)
    return tf.reduce_mean(loss)