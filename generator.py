from tensorflow_core.python.keras.layers.core import Lambda, Dense, Activation
import tensorflow as tf
import numpy as np
import utils
import layers


def generator_model(mapping_layers=8,
                    mapping_fmaps=512,
                    resolution=128,
                    fmap_base=1024,
                    fmap_decay=1.0,
                    fmap_max=512,
                    dtype='float32',
                    num_channels=3):

    # mapping network
    latents = layers.pixel_norm()
    for layer_idx in range(mapping_layers):
        latents = layers.dense(mapping_fmaps)(latents)
        latents = layers.apply_bias()(latents)
        latents = layers.activation()(latents)

    # synthesis network
    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2 * 2 - 2

    def nf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers):
        noise_res = layer_idx // 2 + 2
        shape = [1, 1, 2 ** noise_res, 2 ** noise_res]
        noise_inputs.append(
            tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(),
                            trainable=False))

    def layer_epilogue(x, layer_index):
        x = layers.apply_noise()(x)
        x = layers.apply_bias()(x)
        x = layers.activation()(x)
        x = layers.instance_norm()(x)

        style = layers.index_slice(layer_index)(latents)
        style = layers.dense(x.shape[1] * 2)(style)
        style = layers.apply_bias()(style)

        x = layers.style_mod()([x, style])
        return x

    # early layers
    result = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.ones())
    result = layers.cast(dtype)(result)
    result = layers.tile([tf.shape(latents)[0], 1, 1, 1])(result)
    result = layer_epilogue(result, latents, 0)
    result = layers.conv2d(nf(1), (3, 3))(result)
    result = layer_epilogue(result, latents, 1)

    # Building blocks for remaining layers.
    def block(block_resolution, x):  # res = 3..resolution_log2
        x = layers.upscale2d()(x)
        x = layers.conv2d(nf(block_resolution - 1), (3, 3))(x)
        x = layers.blur2d()(x)
        x = layer_epilogue(x, block_resolution * 2 - 4)

        x = layers.conv2d(nf(block_resolution - 1), (3, 3))(x)
        x = layer_epilogue(x, block_resolution * 2 - 3)
        return x

    def to_rgb(x):  # res = 2..resolution_log2
        x = layers.conv2d(num_channels, (1, 1))(x)
        return layers.apply_bias()(x)

    for res in range(3, resolution_log2 + 1):
        result = block(res, result)
    result = to_rgb(resolution_log2, result)
    return tf.keras.models.Model(inputs=latents, outputs=result)
