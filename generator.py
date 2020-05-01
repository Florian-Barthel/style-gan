import tensorflow as tf
import numpy as np
import layers

epilogue_counter = 0


def generator_model(mapping_layers=8,
                    mapping_fmaps=32,
                    resolution=32,
                    filter_multiplier=1,
                    dtype=tf.dtypes.float32,
                    num_channels=1):

    latents_input = tf.keras.Input(shape=[resolution, 1], dtype=dtype)
    latents = layers.PixelNorm()(latents_input)
    # TODO: only last dense layer has to have filter_number = mapping_fmaps
    for layer_idx in range(mapping_layers):
        latents = layers.dense(mapping_fmaps)(latents)
        latents = layers.activation()(latents)

    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2 - 1
    # num_styles = num_layers * 2

    def number_filters(res):
        res = res / 4
        fn_result = int(resolution / res) * filter_multiplier
        return fn_result

    # noise inputs
    noise_inputs = []
    noise_weights = []
    for layer_idx in range(num_layers):
        noise_res = 2 ** (layer_idx + 2)
        shape = [1, noise_res, noise_res, 1]
        noise_inputs.append(tf.Variable(tf.random.normal(shape), trainable=False, dtype=dtype))
        noise_inputs.append(tf.Variable(tf.random.normal(shape), trainable=False, dtype=dtype))
        noise_weights.append(tf.Variable(tf.zeros([number_filters(noise_res)]), trainable=True, dtype=dtype))
        noise_weights.append(tf.Variable(tf.zeros([number_filters(noise_res)]), trainable=True, dtype=dtype))

    def layer_epilogue(x):
        global epilogue_counter
        x = layers.ApplyNoise(noise=noise_inputs[epilogue_counter], weight=noise_weights[epilogue_counter])(x)
        x = layers.activation()(x)
        x = layers.InstanceNorm()(x)

        style = layers.IndexSlice(epilogue_counter)(latents)
        style = layers.dense(x.shape[3] * 2)(style)
        epilogue_counter += 1
        x = layers.StyleMod()([x, style])
        return x

    # first layer 4 x 4
    batchsize = tf.shape(latents)[0]
    constant = tf.Variable(tf.ones([1, 4, 4, number_filters(4)]), trainable=True, dtype=dtype, name='CONSTANT')
    tile_constant = layers.Tile([batchsize, 1, 1, 1])(constant)
    result = layer_epilogue(tile_constant)
    result = layers.conv2d(number_filters(4), (3, 3))(result)
    result = layer_epilogue(result)

    # remaining layers
    def block(x, layer_index):
        layer_res = 2 ** (layer_index + 2)
        upscale = layers.upscale()(x)
        conv1 = layers.conv2d(number_filters(layer_res), (3, 3))(upscale)
        # TODO: create layers for blur
        # blur = layers.blur2d()(conv1)
        epilogue1 = layer_epilogue(conv1)
        conv2 = layers.conv2d(number_filters(layer_res), (3, 3))(epilogue1)
        epilogue2 = layer_epilogue(conv2)
        return epilogue2

    def to_rgb(x):  # res = 2..resolution_log2
        return layers.conv2d(num_channels, (1, 1))(x)

    # TODO: use Progressive Growing GAN Architecture
    for layer_idx in range(num_layers - 1):
        result = block(result, layer_idx + 1)
    output = to_rgb(result)
    return tf.keras.models.Model(inputs=latents_input, outputs=output)


# model = generator_model()
# tf.keras.utils.plot_model(model, to_file='models/generator_model.png', show_shapes=True, show_layer_names=True, dpi=150)