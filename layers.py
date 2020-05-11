import tensorflow as tf


class Tile(tf.keras.layers.Layer):
    def __init__(self, shape):
        super(Tile, self).__init__()
        self.shape = shape

    def call(self, inputs):
        return tf.tile(inputs, self.shape)


def calc_num_filters(stage, fmap_base):
    return int(fmap_base / (2.0 ** stage))


def conv2d(filters, kernel_size):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  kernel_initializer='random_uniform',
                                  use_bias=True,
                                  activation='linear',
                                  padding='same',
                                  bias_initializer='zeros',
                                  data_format="channels_last")


class Mapping(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_filters):
        super(Mapping, self).__init__()
        self.num_layers = num_layers
        self.layers = []

        for _ in range(self.num_layers):
            self.layers.append(tf.keras.layers.Dense(
                units=num_filters,
                activation='relu',
                use_bias=True,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))

    def call(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


class FirstGenBlock(tf.keras.layers.Layer):
    def __init__(self, fmap_base, type):
        super(FirstGenBlock, self).__init__()
        self.fmap_base = fmap_base
        self.type = type
        self.epilogue_1 = LayerEpilogue(layer_index=0, type=type)
        self.conv = conv2d(filters=calc_num_filters(1, fmap_base), kernel_size=(3, 3))
        self.epilogue_2 = LayerEpilogue(layer_index=1, type=type)

    def build(self, input_shape):
        batchsize = input_shape[0]
        self.constant = tf.Variable(tf.random.normal([1, 4, 4, calc_num_filters(1, self.fmap_base)]), trainable=True,
                                    dtype=self.type)
        self.constant = tf.tile(self.constant, [batchsize, 1, 1, 1])
        super(FirstGenBlock, self).build(input_shape)

    def call(self, latents):
        x = self.epilogue_1([self.constant, latents])
        x = self.conv(x)
        return self.epilogue_2([x, latents])


class GenBlock(tf.keras.layers.Layer):
    def __init__(self, res, fmap_base, type):
        super(GenBlock, self).__init__()
        self.upscale = upscale(2)
        self.conv1 = conv2d(filters=calc_num_filters(res - 1, fmap_base), kernel_size=(3, 3))
        self.epilogue_1 = LayerEpilogue(layer_index=res * 2 - 4, type=type)
        self.conv2 = conv2d(filters=calc_num_filters(res - 1, fmap_base), kernel_size=(3, 3))
        self.epilogue_2 = LayerEpilogue(layer_index=res * 2 - 3, type=type)

    def call(self, inputs):
        x = inputs[0]
        latents = inputs[1]
        x = self.upscale(x)
        x = self.conv1(x)
        x = self.epilogue_1([x, latents])
        x = self.conv2(x)
        return self.epilogue_2([x, latents])


class DiscBlock(tf.keras.layers.Layer):
    def __init__(self, res, fmap_base):
        super(DiscBlock, self).__init__()
        self.conv1 = conv2d(filters=calc_num_filters(res - 1, fmap_base), kernel_size=(3, 3))
        self.activation = activation()
        self.conv2 = conv2d(filters=calc_num_filters(res - 2, fmap_base), kernel_size=(3, 3))
        self.downscale = downscale()

    def call(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return self.downscale(x)


class LastDiscBlock(tf.keras.layers.Layer):
    def __init__(self, fmap_base):
        super(LastDiscBlock, self).__init__()
        self.stddev = MinibatchStdev()
        self.conv1 = conv2d(filters=calc_num_filters(1, fmap_base), kernel_size=(3, 3))
        self.activation1 = activation()
        self.conv2 = conv2d(filters=calc_num_filters(0, fmap_base), kernel_size=(3, 3))
        self.activation2 = activation()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation='linear',
            use_bias=True,
            kernel_initializer='random_uniform',
            bias_initializer='zeros')
        self.activation3 = activation()

    def call(self, x):
        x = self.stddev(x)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.dense(x)
        return self.activation3(x)


class LayerEpilogue(tf.keras.layers.Layer):
    def __init__(self, layer_index, type=tf.dtypes.float32):
        super(LayerEpilogue, self).__init__()
        self.apply_noise = ApplyNoiseWithWeights(layer_index, type)
        self.activation = activation()
        self.instance_norm = InstanceNorm()
        self.slice = IndexSlice(layer_index)
        self.style_mod = StyleMod()

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(
            units=input_shape[0][3] * 2,
            activation='linear',
            use_bias=True,
            kernel_initializer='random_uniform',
            bias_initializer='zeros')
        super(LayerEpilogue, self).build(input_shape)

    def call(self, inputs):
        x = inputs[0]
        style = inputs[1]
        x = self.apply_noise(x)
        x = self.activation(x)
        x = self.instance_norm(x)
        style = self.slice(style)
        style = self.dense(style)
        return self.style_mod([x, style])


class ToRGB(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ToRGB, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=num_channels,
                                           kernel_size=(1, 1),
                                           kernel_initializer='random_uniform',
                                           use_bias=True,
                                           activation='linear',
                                           padding='same',
                                           bias_initializer='zeros',
                                           data_format="channels_last")

    def call(self, x):
        y = self.conv(x)
        return y


class FromRGB(tf.keras.layers.Layer):
    def __init__(self, res, fmap_base):
        super(FromRGB, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=calc_num_filters(res - 1, fmap_base=fmap_base),
                                           kernel_size=(1, 1),
                                           kernel_initializer='random_uniform',
                                           use_bias=True,
                                           activation='relu',
                                           padding='same',
                                           bias_initializer='zeros',
                                           data_format="channels_last")

    def call(self, x):
        return self.conv(x)


def activation():
    return tf.keras.layers.Activation('relu')


def upscale(faktor=2):
    return tf.keras.layers.UpSampling2D(size=(faktor, faktor), data_format='channels_last', interpolation='bilinear')


def downscale():
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')


class ApplyNoiseWithWeights(tf.keras.layers.Layer):
    def __init__(self, layer_idx, type=tf.float32):
        super(ApplyNoiseWithWeights, self).__init__()
        noise_res = layer_idx // 2 + 2
        self.noise_shape = [1, 2 ** noise_res, 2 ** noise_res, 1]
        self.type = type
        self.noise = tf.Variable(tf.random.normal(self.noise_shape), trainable=False, dtype=self.type)
        self.noise_weight = None

    def build(self, input_shape):
        self.noise_weight = self.add_weight(name='noise_weight',
                                            shape=[input_shape[3]],
                                            initializer='uniform',
                                            trainable=True)
        super(ApplyNoiseWithWeights, self).build(input_shape)

    def call(self, inputs):
        return inputs + self.noise * self.noise_weight


class InstanceNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(InstanceNorm, self).__init__()
        self.epsilon = tf.Variable(1e-8, trainable=False)

    def call(self, x):
        x -= tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + self.epsilon)
        return x


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = tf.Variable(1e-8, trainable=False)

    def call(self, inputs):
        return inputs / tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + self.epsilon)


class StyleMod(tf.keras.layers.Layer):
    def __init__(self):
        super(StyleMod, self).__init__()

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
class MinibatchStdev(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        squ_diffs = tf.math.square(inputs - mean)
        mean_sq_diff = tf.reduce_mean(squ_diffs, axis=0, keepdims=True)
        mean_sq_diff += 1e-8
        stdev = tf.math.sqrt(mean_sq_diff)
        mean_pix = tf.reduce_mean(stdev, keepdims=True)
        shape = tf.shape(inputs)
        output = tf.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        combined = tf.concat([inputs, output], axis=-1)
        return combined

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)
