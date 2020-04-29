import tensorflow as tf


def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]


def D_logistic_simplegp(fake_images_out, real_scores_out, fake_scores_out, opt, reals, r1_gamma=10.0, r2_gamma=0.0): # pylint: disable=unused-argument

    loss = tf.nn.softplus(fake_scores_out)  # -log(1 - logistic(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type

    if r1_gamma != 0.0:
        with tf.name_scope('R1Penalty'):
            real_loss = opt.apply_loss_scaling(tf.reduce_sum(real_scores_out))
            real_grads = opt.undo_loss_scaling(fp32(tf.gradients(real_loss, [reals])[0]))
            r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        loss += r1_penalty * (r1_gamma * 0.5)

    if r2_gamma != 0.0:
        with tf.name_scope('R2Penalty'):
            fake_loss = opt.apply_loss_scaling(tf.reduce_sum(fake_scores_out))
            fake_grads = opt.undo_loss_scaling(fp32(tf.gradients(fake_loss, [fake_images_out])[0]))
            r2_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1,2,3])
        loss += r2_penalty * (r2_gamma * 0.5)
    return loss
