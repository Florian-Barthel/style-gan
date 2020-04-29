import tensorflow as tf
import utils


def cast(dtype):
    def func(x):
        tf.cast(x, dtype)
    return tf.keras.layers.Lambda(func)


def tile(shape):
    def func(x):
        tf.tile(x, shape)
    return tf.keras.layers.Lambda(func)


def conv2d(filters, kernel_size):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  kernel_initializer='random_uniform',
                                  use_bias=False,
                                  activation='linear')


def dense(units):
    return tf.keras.layers.Dense(units=units, activation=None, use_bias=False)


def activation():
    return tf.keras.layers.Activation('relu')


def apply_bias():
    def func(x):
        lrmul = 1
        b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
        b = tf.cast(b, x.dtype)
        if len(x.shape) == 2:
            return x + b
        return x + tf.reshape(b, [1, -1, 1, 1])
    return tf.keras.layers.Lambda(func)


def downscale2d():
    factor = 2
    @tf.custom_gradient
    def func(x):
        y = utils.downscale2d(x, factor)

        @tf.custom_gradient
        def grad(dy):
            dx = utils.upscale2d(dy, factor, gain=1 / factor ** 2)
            return dx, lambda ddx: utils.downscale2d(ddx, factor)

        return y, grad
    return tf.keras.layers.Lambda(func)

def upscale2d():
    factor = 2

    @tf.custom_gradient
    def func(x):
        y = utils.upscale2d(x, factor)

        @tf.custom_gradient
        def grad(dy):
            dx = utils.downscale2d(dy, factor, gain=factor ** 2)
            return dx, lambda ddx: utils.upscale2d(ddx, factor)

        return y, grad

    return tf.keras.layers.Lambda(func)


def apply_noise():
    def func(x):
        noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        weight = tf.get_variable('weight', shape=[x.shape[1].value], initializer=tf.initializers.zeros())
        return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1])
    return tf.keras.layers.Lambda(func)


def instance_norm():
    def func(x):
        epsilon = 1e-8
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2, 3], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        return x

    return tf.keras.layers.Lambda(func)


def pixel_norm(x):
    def func():
        epsilon = 1e-8
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

    return tf.keras.layers.Lambda(func)


def blur2d():
    f = [1, 2, 1]
    normalize = True

    @tf.custom_gradient
    def func(x):
        y = utils.blur2d(x, f, normalize)

        @tf.custom_gradient
        def grad(dy):
            dx = utils.blur2d(dy, f, normalize, flip=True)
            return dx, lambda ddx: utils.blur2d(ddx, f, normalize)

        return y, grad

    return tf.keras.layers.Lambda(func)


def minibatch_stddev():
    def func(x):
        group_size = 4
        num_new_features = 1
        group_size = tf.minimum(group_size,
                                tf.shape(x)[0])  # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape  # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, num_new_features, s[1] // num_new_features, s[2], s[
            3]])  # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)  # [GMncHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMncHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)  # [MncHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)  # [MncHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)  # [Mn111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[2])  # [Mn11] Split channels into c channel groups
        y = tf.cast(y, x.dtype)  # [Mn11]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])  # [NnHW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)  # [NCHW]  Append as new fmap.
    return tf.keras.layers.Lambda(func)


def style_mod():
    def func(x, style):
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        return x * (style[:, 0] + 1) + style[:, 1]

    return tf.keras.layers.Lambda(func)


def style_mod():
    def func(x, style):
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        return x * (style[:, 0] + 1) + style[:, 1]

    return tf.keras.layers.Lambda(func)


def index_slice(layer_idx):
    def func(x):
        return x[:, layer_idx]
    return tf.keras.layers.Lambda(func)
