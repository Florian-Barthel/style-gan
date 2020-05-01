import tensorflow as tf
import utils


def cast(dtype, shape):
    def func(x):
        tf.cast(x, dtype)

    return tf.keras.layers.Lambda(func, input_shape=shape, output_shape=shape)


class Tile(tf.keras.layers.Layer):
    def __init__(self, shape):
        super(Tile, self).__init__()
        self.shape = shape

    def call(self, inputs):
        return tf.tile(inputs, self.shape)


def conv2d(filters, kernel_size):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  kernel_initializer='random_uniform',
                                  use_bias=True,
                                  activation='linear',
                                  padding='same')


def dense(units):
    return tf.keras.layers.Dense(units=units, activation=None, use_bias=True)


def activation():
    return tf.keras.layers.Activation('relu')


def upscale():
    return tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')


def apply_noise(weight, noise):
    def func(x):
        return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1])

    return tf.keras.layers.Lambda(func)


class ApplyNoise(tf.keras.layers.Layer):
    def __init__(self, noise, weight):
        super(ApplyNoise, self).__init__()
        self.weight = weight
        self.noise = noise

    def call(self, inputs):
        return inputs + self.noise * tf.reshape(tf.cast(self.weight, inputs.dtype), [1, 1, 1, -1])


class InstanceNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(InstanceNorm, self).__init__()

    def call(self, inputs):
        epsilon = 1e-8
        orig_dtype = inputs.dtype
        x = tf.cast(inputs, tf.float32)
        x -= tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x = x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=[2, 3], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        return x


def pixel_norm():
    def func(x):
        epsilon = 1e-8
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

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


class StyleMod(tf.keras.layers.Layer):
    def __init__(self):
        super(StyleMod, self).__init__()

    def call(self, inputs):
        x = inputs[0]
        style = inputs[1]
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        return x * (style[:, 0] + 1) + style[:, 1]
        return x




def index_slice(layer_idx):
    def func(x):
        return x[:, layer_idx]

    return tf.keras.layers.Lambda(func)


class IndexSlice(tf.keras.layers.Layer):
    def __init__(self, index):
        super(IndexSlice, self).__init__()
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index]
