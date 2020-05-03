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


class StyleMod2(tf.keras.layers.Layer):
    def __init__(self):
        super(StyleMod2, self).__init__()

    def call(self, inputs):
        x = inputs[0]
        style = inputs[1]
        style = tf.reshape(style, [-1, 2, 1, 1, x.shape[3]])

        style_s = (style[:, 0, :, :, :] + 1)
        style_b = style[:, 1, :, :, :]
        return x * style_s + style_b


class IndexSlice(tf.keras.layers.Layer):
    def __init__(self, index):
        super(IndexSlice, self).__init__()
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index]


# src: https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/
# mini-batch standard deviation layer
class MinibatchStdev(tf.keras.layers.Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = tf.math.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = tf.reduce_mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = tf.math.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = tf.reduce_mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = tf.shape(inputs)
        output = tf.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = tf.concat([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)

