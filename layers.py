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


def downscale():
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')


def apply_noise(weight, noise):
    def func(x):
        return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1])

    return tf.keras.layers.Lambda(func)


# class ApplyNoise(tf.keras.layers.Layer):
#     def __init__(self, noise, weight):
#         super(ApplyNoise, self).__init__()
#         self.weight = weight
#         self.noise = noise
#
#     def call(self, inputs):
#         return inputs + self.noise * tf.reshape(tf.cast(self.weight, inputs.dtype), [1, 1, 1, -1])


class ApplyNoiseWithWeights(tf.keras.layers.Layer):
    def __init__(self, noise):
        super(ApplyNoiseWithWeights, self).__init__()
        self.noise = noise

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.noise_weight = self.add_weight(name='noise_weight',
                                            shape=(input_shape[3]),
                                            initializer='zeros',
                                            trainable=True)
        super(ApplyNoiseWithWeights, self).build(input_shape)

    def call(self, inputs):
        return inputs + self.noise * self.noise_weight


class InstanceNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(InstanceNorm, self).__init__()

    def call(self, inputs):
        epsilon = 1e-8
        orig_dtype = inputs.dtype
        x = tf.cast(inputs, tf.float32)
        x -= tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x = x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        return x


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def call(self, inputs):
        epsilon = 1e-8
        epsilon = tf.constant(epsilon, dtype=inputs.dtype, name='epsilon')
        return inputs / tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + epsilon)


# def blur2d():
#     f = [1, 2, 1]
#     normalize = True
#
#     @tf.custom_gradient
#     def func(x):
#         y = utils.blur2d(x, f, normalize)
#
#         @tf.custom_gradient
#         def grad(dy):
#             dx = utils.blur2d(dy, f, normalize, flip=True)
#             return dx, lambda ddx: utils.blur2d(ddx, f, normalize)
#
#         return y, grad
#
#     return tf.keras.layers.Lambda(func)


class StyleMod(tf.keras.layers.Layer):
    def __init__(self):
        super(StyleMod, self).__init__()

    def call(self, inputs):
        x = inputs[0]
        style = inputs[1]
        # style = tf.reshape(style, [-1, x.shape[3]] + [1] * (len(x.shape) - 2), 2)
        style_mean = tf.math.reduce_mean(style)
        style_stddev = tf.math.reduce_std(style)
        return style_stddev * x + style_mean


class IndexSlice(tf.keras.layers.Layer):
    def __init__(self, index):
        super(IndexSlice, self).__init__()
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index]
