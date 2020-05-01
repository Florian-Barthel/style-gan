import tensorflow as tf
import utils
import numpy as np
import layers


def discriminator_model(
        resolution=32,
        dtype='float32',
        filter_multiplier=1,
        number_of_channels=1):

    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2
    input_image = tf.keras.Input(shape=[resolution, resolution, number_of_channels], dtype=dtype)

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
        if layer_res >= 8:
            conv1 = layers.conv2d(filters=number_filters(layer_res), kernel_size=(3, 3))(x)
            act = layers.activation()(conv1)
            blur = layers.blur2d()(act)
            conv2 = layers.conv2d(filters=number_filters(half_layer_res), kernel_size=(3, 3))(blur)
            block_result = layers.downscale()(conv2)
        else:
            conv1 = layers.conv2d(filters=number_filters(layer_res), kernel_size=(3, 3))(x)
            act1 = layers.activation()(conv1)
            dense1 = layers.dense(units=number_filters(half_layer_res))(act1)
            act2 = layers.activation()(dense1)
            dense2 = layers.dense(units=1)(act2)
            block_result = layers.activation()(dense2)
        return block_result

    # TODO: use Progressive Growing GAN Architecture
    result = from_rgb(input_image, resolution)
    for layer_idx in range(num_layers, 0, -1):
        result = block(result, layer_idx)
    output = block(result, 0)

    return tf.keras.models.Model(inputs=input_image, outputs=output)


# model = discriminator_model()
# tf.keras.utils.plot_model(model, to_file='models/discriminator_model.png', show_shapes=True, show_layer_names=True, dpi=150)
