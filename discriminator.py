import tensorflow as tf
import utils
import numpy as np
import layers


def discriminator_model(
        resolution=64,
        dtype='float32',
        filter_multiplier=1):

    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2 - 2
    input_image = tf.keras.Input(shape=[resolution, resolution, 3], dtype=dtype)

    def number_filters(res):
        res = res / 4
        fn_result = int(resolution / res) * filter_multiplier
        return fn_result

    def from_rgb(x, res):
        conv = layers.conv2d(filters=number_filters(res), kernel_size=(1, 1))(x)
        act = layers.activation()(conv)
        return act

    def block(x, layer_index):
        layer_res = 2 ** (layer_index + 2)
        half_layer_res = layer_res / 2
        print('BLOCK {} x {}'.format(layer_res, layer_res))
        if layer_res >= 8:
            conv1 = layers.conv2d(filters=number_filters(layer_res), kernel_size=(3, 3))(x)
            print('\tconv2d (1) filters: {}, kernel: (3, 3), shape: {}'.format(number_filters(layer_res), conv1.shape))
            act = layers.activation()(conv1)
            blur = layers.blur2d()(act)
            conv2 = layers.conv2d(filters=number_filters(half_layer_res), kernel_size=(3, 3))(blur)
            print('\tconv2d (1) filters: {}, kernel: (3, 3), shape: {}'.format(number_filters(half_layer_res), conv2.shape))
            block_result = layers.downscale()(conv2)
        else:
            conv1 = layers.conv2d(filters=number_filters(layer_res), kernel_size=(3, 3))(x)
            print('\tconv2d (1) filters: {}, kernel: (3, 3), shape: {}'.format(number_filters(layer_res), conv1.shape))
            act1 = layers.activation()(conv1)
            dense1 = layers.dense(units=number_filters(half_layer_res))(act1)
            print('\tdense (1) filters: {}, shape: {}'.format(number_filters(half_layer_res), dense1.shape))
            act2 = layers.activation()(dense1)
            dense2 = layers.dense(units=1)(act2)
            print('\tdense (1) filters: {}, shape: {}'.format(1, dense2.shape))
            block_result = layers.activation()(dense2)
        return block_result

    # TODO: use Progressive Growing GAN Architecture
    result = from_rgb(input_image, resolution)
    for layer_idx in range(num_layers, 0, -1):
        result = block(result, layer_idx)
    output = block(result, 0)

    return tf.keras.models.Model(inputs=input_image, outputs=output)


model = discriminator_model()
tf.keras.utils.plot_model(model, to_file='models/discriminator_model.png', show_shapes=True, show_layer_names=True, dpi=150)
