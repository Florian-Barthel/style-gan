import tensorflow as tf
import numpy as np
import layers

epilogue_counter = 0


def generator_model(mapping_layers=8,
                    mapping_fmaps=64,
                    resolution=64,
                    filter_multiplier=1,
                    dtype=tf.dtypes.float32,
                    num_channels=3):

    latents_input = tf.keras.Input(shape=[64, 1], dtype=dtype)
    latents = layers.PixelNorm()(latents_input)
    for layer_idx in range(mapping_layers):
        latents = layers.dense(mapping_fmaps)(latents)
        latents = layers.activation()(latents)

    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2 - 1

    def number_filters(res):
        res = res / 4
        fn_result = int(resolution / res) * filter_multiplier
        return fn_result

    # noise inputs
    noise_inputs = []
    noise_weights = []
    print('noise:')
    for layer_idx in range(num_layers):
        noise_res = 2 ** (layer_idx + 2)
        shape = [1, noise_res, noise_res, 1]
        noise_inputs.append(tf.Variable(tf.random.normal(shape), trainable=False, dtype=dtype))
        print('noise(1) layer {}, shape: (None, {}, {}, 1)'.format(layer_idx, noise_res, noise_res))
        noise_inputs.append(tf.Variable(tf.random.normal(shape), trainable=False, dtype=dtype))
        print('noise(2) layer {}, shape: (None, {}, {}, 1)'.format(layer_idx, noise_res, noise_res))
        noise_weights.append(tf.Variable(tf.zeros([number_filters(noise_res)]), trainable=True, dtype=dtype))
        print('noise-weight(1) layer {}, shape: ({})'.format(layer_idx, number_filters(noise_res)))
        noise_weights.append(tf.Variable(tf.zeros([number_filters(noise_res)]), trainable=True, dtype=dtype))
        print('noise-weight(2) layer {}, shape: ({})'.format(layer_idx, number_filters(noise_res)))
    print('')
    print('')

    def layer_epilogue(x):
        global epilogue_counter
        print('\tepilogue: (epilogue_index: {})'.format(epilogue_counter))
        x = layers.ApplyNoise(noise=noise_inputs[epilogue_counter], weight=noise_weights[epilogue_counter])(x)
        print('\t\tapply noise layer {}, shape: {}'.format(epilogue_counter, x.shape))
        x = layers.activation()(x)
        print('\t\tactivation layer {}, shape: {}'.format(epilogue_counter, x.shape))
        x = layers.InstanceNorm()(x)
        print('\t\tinstance norm layer {}, shape: {}'.format(epilogue_counter, x.shape))

        style = layers.IndexSlice(epilogue_counter)(latents)
        print('\t\tstyle slice layer {}, shape: {}'.format(epilogue_counter, style.shape))
        style = layers.dense(x.shape[3])(style)
        print('\t\tstyle dense layer {}, filters: {}, shape: {}'.format(epilogue_counter, x.shape[3], style.shape))
        epilogue_counter += 1
        x = layers.StyleMod()([x, style])
        return x

    # first layer
    print('BLOCK 4 x 4')
    batchsize = tf.shape(latents)[0]
    constant = tf.Variable(tf.ones([1, 4, 4, number_filters(4)]), trainable=True, dtype=dtype, name='CONSTANT')
    print('\tconstant shape: (1, 4, 4, {})'.format(number_filters(4)))
    tile_constant = layers.Tile([batchsize, 1, 1, 1])(constant)
    print('\ttile constant shape:  {}'.format(tile_constant.shape))
    result = layer_epilogue(tile_constant)
    result = layers.conv2d(number_filters(4), (3, 3))(result)
    print('\tconv2d filters: {}, kernel: (3, 3)'.format(number_filters(4)))
    result = layer_epilogue(result)

    # remaining layers
    def block(x, layer_index):
        layer_res = 2 ** (layer_index + 2)
        print('BLOCK {} x {}'.format(layer_res, layer_res))
        upscale = layers.upscale()(x)
        conv1 = layers.conv2d(number_filters(layer_res), (3, 3))(upscale)
        # TODO: create layers for blur
        blur = layers.blur2d()(conv1)
        epilogue1 = layer_epilogue(blur)
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


model = generator_model()
tf.keras.utils.plot_model(model, to_file='models/generator_model.png', show_shapes=True, show_layer_names=True, dpi=150)