import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, MaxPooling2D, UpSampling2D, LeakyReLU
import config


def calc_num_filters(stage, fmap_base, fmap_max=config.fmap_max, fmap_decay=1):
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


def upscale(factor=2):
    return UpSampling2D(size=(factor, factor), data_format='channels_last', interpolation='bilinear')


def downscale():
    return MaxPooling2D(pool_size=(2, 2), data_format='channels_last')


def activation():
    return LeakyReLU(alpha=0.2)


class CustomDense(Layer):
    def __init__(self, units, use_wscale, lr_mul=1, type=tf.float32, gain=np.sqrt(2)):
        super(CustomDense, self).__init__()
        self.units = units
        self.lr_mul = lr_mul
        self.type = type
        self.use_wscale = use_wscale
        self.gain = gain
        self.weight = None
        self.init_std = None
        self.runtime_coef = None

    def build(self, input_shape):
        fan_in = input_shape[-1]
        he_std = self.gain / np.sqrt(fan_in)
        if self.use_wscale:
            self.init_std = 1.0 / self.lr_mul
            self.runtime_coef = he_std * self.lr_mul
        else:
            self.init_std = he_std / self.lr_mul
            self.runtime_coef = self.lr_mul

        self.weight = tf.Variable(tf.random.normal([input_shape[-1], self.units], stddev=self.init_std),
                                  trainable=True,
                                  dtype=self.type)
        super(CustomDense, self).build(input_shape)

    def call(self, x, **kwargs):
        w = self.weight * self.runtime_coef
        return tf.matmul(x, w)


class CustomConv2d(Layer):
    def __init__(self, filters, kernel, lr_mul=1, type=tf.float32, use_wscale=True, gain=np.sqrt(2)):
        super(CustomConv2d, self).__init__()
        self.filters = filters
        self.kernel = kernel
        self.lr_mul = lr_mul
        self.type = type
        self.use_wscale = use_wscale
        self.gain = gain
        self.weight = None
        self.init_std = None
        self.runtime_coef = None

    def build(self, input_shape):
        fan_in = self.kernel * self.kernel * input_shape[-1]
        he_std = self.gain / np.sqrt(fan_in)
        if self.use_wscale:
            self.init_std = 1.0 / self.lr_mul
            self.runtime_coef = he_std * self.lr_mul
        else:
            self.init_std = he_std / self.lr_mul
            self.runtime_coef = self.lr_mul

        self.weight = tf.Variable(tf.random.normal([self.kernel, self.kernel, input_shape[-1], self.filters],
                                                   stddev=self.init_std),
                                  trainable=True,
                                  dtype=self.type)
        super(CustomConv2d, self).build(input_shape)

    def call(self, x, **kwargs):
        w = self.weight * self.runtime_coef
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')


class Mapping(Layer):
    def __init__(self, num_layers, num_filters, num_style_layers, lr_mul, type, use_wscale):
        super(Mapping, self).__init__()
        self.num_layers = num_layers
        self.num_style_layers = num_style_layers
        self.num_filters = num_filters
        self.layers = []
        self.bias = []
        self.activation = []

        for _ in range(self.num_layers):
            self.layers.append(CustomDense(self.num_filters, lr_mul=lr_mul, use_wscale=use_wscale))
            self.bias.append(ApplyBias(lr_mul, type))
            self.activation.append(activation())

    def call(self, x, **kwargs):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.bias[i](x)
            x = self.activation[i](x)
        x = tf.tile(tf.expand_dims(x, axis=1), [1, self.num_style_layers, 1])
        return x


class GetDlatentAvg(Layer):
    def __init__(self, dlatent_avg_beta):
        super(GetDlatentAvg, self).__init__()
        self.dlatent_avg = tf.Variable(tf.zeros([config.dlatent_size]), trainable=False)
        self.dlatent_avg_beta = dlatent_avg_beta

    def call(self, inputs, **kwargs):
        x = inputs[0]
        is_validation = inputs[1]
        if is_validation == 0.0:
            batch_avg = tf.reduce_mean(x[:, 0], axis=0)
            self.dlatent_avg.assign(batch_avg + (self.dlatent_avg - batch_avg) * self.dlatent_avg_beta)
        return [x, self.dlatent_avg]


class ApplyTruncation(Layer):
    def __init__(self, num_style_layers, truncation_cutoff, truncation_psi):
        super(ApplyTruncation, self).__init__()
        layer_idx = np.arange(num_style_layers)[np.newaxis, :, np.newaxis]
        ones = np.ones(layer_idx.shape, dtype=np.float32)
        self.coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)

    def call(self, inputs, **kwargs):
        dlatent = inputs[0]
        dlatent_avg = inputs[1]
        return dlatent_avg + (dlatent - dlatent_avg) * self.coefs


class ApplyBias(Layer):
    def __init__(self, lr_mul=1, type=tf.float32):
        super(ApplyBias, self).__init__()
        self.lr_mul = lr_mul
        self.type = type
        self.bias = None

    def build(self, input_shape):
        self.bias = tf.Variable(tf.zeros([input_shape[-1]]),
                                trainable=True,
                                dtype=self.type)
        super(ApplyBias, self).build(input_shape)

    def call(self, x, **kwargs):
        return x + self.bias * self.lr_mul


class FirstGenBlock(Layer):
    def __init__(self, fmap_base, type, gain, use_wscale):
        super(FirstGenBlock, self).__init__()
        self.fmap_base = fmap_base
        self.type = type
        self.epilogue_1 = LayerEpilogue(layer_index=0, type=type, use_wscale=use_wscale)
        self.conv = CustomConv2d(filters=calc_num_filters(1, fmap_base), kernel=3, gain=gain, use_wscale=use_wscale)
        self.epilogue_2 = LayerEpilogue(layer_index=1, type=type, use_wscale=use_wscale)
        self.constant = tf.Variable(tf.ones([1, 4, 4, calc_num_filters(1, self.fmap_base)]),
                                    trainable=True,
                                    dtype=self.type)

    def call(self, latents, **kwargs):
        x = self.epilogue_1([tf.tile(self.constant, [tf.shape(latents)[0], 1, 1, 1]), latents])
        x = self.conv(x)
        return self.epilogue_2([x, latents])


class GenBlock(Layer):
    def __init__(self, res, fmap_base, type, use_wscale, gain):
        super(GenBlock, self).__init__()
        self.upscale = upscale(2)
        self.conv1 = CustomConv2d(filters=calc_num_filters(res - 1, fmap_base), kernel=3, gain=gain)
        self.blur = Blur2d()
        self.epilogue_1 = LayerEpilogue(layer_index=res * 2 - 4, type=type, use_wscale=use_wscale)
        self.conv2 = CustomConv2d(filters=calc_num_filters(res - 1, fmap_base), kernel=3, gain=gain,
                                  use_wscale=use_wscale)
        self.epilogue_2 = LayerEpilogue(layer_index=res * 2 - 3, type=type, use_wscale=use_wscale)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        latents = inputs[1]
        x = self.upscale(x)
        x = self.conv1(x)
        x = self.blur(x)
        x = self.epilogue_1([x, latents])
        x = self.conv2(x)
        return self.epilogue_2([x, latents])


class DiscBlock(Layer):
    def __init__(self, res, fmap_base, gain, use_wscale):
        super(DiscBlock, self).__init__()
        self.conv1 = CustomConv2d(filters=calc_num_filters(res - 1, fmap_base), kernel=3, gain=gain,
                                  use_wscale=use_wscale)
        self.apply_bias1 = ApplyBias()
        self.activation1 = activation()
        self.blur = Blur2d()
        self.conv2 = CustomConv2d(filters=calc_num_filters(res - 2, fmap_base), kernel=3, gain=gain,
                                  use_wscale=use_wscale)
        self.downscale = downscale()
        self.apply_bias2 = ApplyBias()
        self.activation2 = activation()

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.apply_bias1(x)
        x = self.activation1(x)
        x = self.blur(x)
        x = self.conv2(x)
        x = self.downscale(x)
        x = self.apply_bias2(x)
        x = self.activation2(x)
        return x


class LastDiscBlock(Layer):
    def __init__(self, fmap_base, gain, use_wscale):
        super(LastDiscBlock, self).__init__()
        self.stddev = MinibatchStdev(group_size=4, num_new_features=1)
        self.conv1 = CustomConv2d(filters=calc_num_filters(1, fmap_base), kernel=3, gain=gain, use_wscale=use_wscale)
        self.apply_bias1 = ApplyBias()
        self.activation1 = activation()
        self.dense1 = CustomDense(units=calc_num_filters(0, fmap_base), gain=gain, use_wscale=use_wscale)
        self.apply_bias2 = ApplyBias()
        self.activation2 = activation()
        self.dense2 = CustomDense(units=1, gain=1, use_wscale=use_wscale)
        self.apply_bias3 = ApplyBias()

    def call(self, x, **kwargs):
        x = self.stddev(x)
        x = self.conv1(x)
        x = self.apply_bias1(x)
        x = self.activation1(x)
        x = self.dense1(x)
        x = self.apply_bias2(x)
        x = self.activation2(x)
        x = self.dense2(x)
        x = self.apply_bias3(x)
        return x


class LayerEpilogue(Layer):
    def __init__(self, layer_index, use_wscale, type=tf.float32):
        super(LayerEpilogue, self).__init__()
        self.layer_index = layer_index
        self.apply_noise = ApplyNoise(layer_index, type)
        self.apply_bias = ApplyBias()
        self.activation = activation()
        self.instance_norm = InstanceNorm()
        self.style_mod = StyleMod(use_wscale=use_wscale)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        style = inputs[1]
        x = self.apply_noise(x)
        x = self.apply_bias(x)
        x = self.activation(x)
        x = self.instance_norm(x)
        style = style[:, self.layer_index]
        return self.style_mod([x, style])


class StyleMod(Layer):
    def __init__(self, use_wscale):
        super(StyleMod, self).__init__()
        self.dense = None
        self.apply_bias = ApplyBias()
        self.use_wscale = use_wscale

    def build(self, input_shape):
        self.dense = CustomDense(units=input_shape[0][3] * 2, gain=1, use_wscale=self.use_wscale)
        super(StyleMod, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        style = inputs[1]
        style = tf.expand_dims(style, axis=1)
        style = self.dense(style)
        style = self.apply_bias(style)
        style = tf.reshape(style, [-1, 2, 1, 1, x.shape[3]])
        style_s = style[:, 0, :, :, :] + 1
        style_b = style[:, 1, :, :, :]
        return x * style_s + style_b


class ToRGB(Layer):
    def __init__(self, num_channels, use_wscale):
        super(ToRGB, self).__init__()
        self.conv = CustomConv2d(filters=num_channels, kernel=1, gain=1, use_wscale=use_wscale)
        self.apply_bias = ApplyBias()

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.apply_bias(x)
        return x


class FromRGB(Layer):
    def __init__(self, res, fmap_base, gain, use_wscale):
        super(FromRGB, self).__init__()
        self.conv = CustomConv2d(filters=calc_num_filters(res - 1, fmap_base=fmap_base), kernel=1, gain=gain,
                                 use_wscale=use_wscale)
        self.apply_bias = ApplyBias()
        self.activation = activation()

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.apply_bias(x)
        x = self.activation(x)
        return x


class ApplyNoise(Layer):
    def __init__(self, layer_idx, type=tf.float32):
        super(ApplyNoise, self).__init__()
        self.noise_res = layer_idx // 2 + 2
        # self.noise_shape = [1, 2 ** noise_res, 2 ** noise_res, 1]
        # self.noise = tf.Variable(tf.random.normal(self.noise_shape), trainable=False, dtype=type)
        self.weight = None

    def build(self, input_shape):
        self.weight = self.add_weight(name='noise_weight',
                                      shape=[input_shape[3]],
                                      initializer='zeros',
                                      trainable=True)
        super(ApplyNoise, self).build(input_shape)

    def call(self, inputs, **kwargs):
        noise_shape = [tf.shape(inputs)[0], 2 ** self.noise_res, 2 ** self.noise_res, 1]
        return inputs + tf.random.normal(noise_shape) * self.weight


class InstanceNorm(Layer):
    def __init__(self):
        super(InstanceNorm, self).__init__()
        self.epsilon = tf.Variable(1e-8, dtype=tf.float32, trainable=False)

    def call(self, x, **kwargs):
        x -= tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + self.epsilon)
        return x


# for latent vector only
class PixelNorm(Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = tf.Variable(1e-8, dtype=tf.float32, trainable=False)

    def call(self, inputs, **kwargs):
        return inputs / tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + self.epsilon)


class MinibatchStdev(Layer):
    def __init__(self, num_new_features, group_size):
        super(MinibatchStdev, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, x, **kwargs):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        s = x.shape #NHWC
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3] // self.num_new_features, self.num_new_features]) #GMHWcn
        y = tf.cast(y, tf.float32) #GMHWcn
        y -= tf.reduce_mean(y, axis=0, keepdims=True) #GMHWcn
        y = tf.reduce_mean(tf.square(y), axis=0) #MHWcn
        y = tf.sqrt(y + 1e-8) #MHWcn
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True) #MHWcn
        y = tf.reduce_mean(y, axis=3) #MHWn
        y = tf.cast(y, x.dtype) #MHWn
        y = tf.tile(y, [group_size, s[1], s[2], 1]) #NHWn
        return tf.concat([x, y], axis=-1) #NHWC


class Blur2d(Layer):
    def __init__(self):
        super(Blur2d, self).__init__()
        self.filter = None

    def build(self, input_shape):
        self.filter = [1, 2, 1]
        self.filter = np.array(self.filter, dtype=np.float32)
        self.filter = self.filter[:, np.newaxis] * self.filter[np.newaxis, :]
        self.filter /= np.sum(self.filter)
        self.filter = self.filter[:, :, np.newaxis, np.newaxis]
        self.filter = np.tile(self.filter, [1, 1, int(input_shape[3]), 1])
        super(Blur2d, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.nn.depthwise_conv2d(inputs, self.filter, [1, 1, 1, 1], padding='SAME', data_format='NHWC')
