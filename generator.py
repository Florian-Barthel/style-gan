import tensorflow as tf
import numpy as np
import layers


def generator_model(mapping_layers=8,
                    mapping_fmaps=32,
                    resolution=32,
                    dtype=tf.dtypes.float32,
                    num_channels=1,
                    fmap_base=32):

    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2 * 2 - 2

    latents_input = tf.keras.Input(shape=[resolution, 1], dtype=dtype)
    latents = layers.PixelNorm()(latents_input)
    lod_input = tf.keras.Input(shape=[1], dtype=dtype, name='lod')

    # quick fix to remove the batch dimension
    lod_in = tf.reduce_max(lod_input)

    # TODO: only last dense layer has to have filter_number = mapping_fmaps
    for layer_idx in range(mapping_layers):
        latents = layers.dense(mapping_fmaps)(latents)
        latents = layers.activation()(latents)

    def num_filters(stage):
        return int(fmap_base / (2.0 ** stage))

    noise_inputs = []
    for layer_idx in range(num_layers):
        res = layer_idx // 2 + 2
        shape = [1, 2 ** res, 2 ** res, 1]
        noise_inputs.append(tf.Variable(tf.random.normal(shape), trainable=False, dtype=dtype))

    def layer_epilogue(x, layer_index):
        x = layers.ApplyNoiseWithWeights(noise=noise_inputs[layer_index])(x)
        x = layers.activation()(x)
        x = layers.InstanceNorm()(x)
        style = layers.IndexSlice(layer_index)(latents)
        style = layers.dense(x.shape[3] * 2)(style)
        x = layers.StyleMod2()([x, style])
        return x

    def block(x, res):
        upscale = layers.upscale()(x)
        conv1 = layers.conv2d(num_filters(res - 1), (3, 3))(upscale)
        # TODO: create layers for blur
        epilogue1 = layer_epilogue(conv1, res * 2 - 4)
        conv2 = layers.conv2d(num_filters(res - 1), (3, 3))(epilogue1)
        epilogue2 = layer_epilogue(conv2, res * 2 - 3)
        return epilogue2

    def to_rgb(x):
        return layers.conv2d(num_channels, (1, 1))(x)

    # first layer 4 x 4
    batchsize = tf.shape(latents)[0]
    constant = tf.Variable(tf.random.normal([1, 4, 4, num_filters(1)]), trainable=True, dtype=dtype, name='CONSTANT')
    tile_constant = layers.Tile([batchsize, 1, 1, 1])(constant)
    result = layer_epilogue(tile_constant, 0)
    result = layers.conv2d(num_filters(1), (3, 3))(result)
    result = layer_epilogue(result, 1)

    images_skip = to_rgb(result)
    for res in range(3, resolution_log2 + 1):
        lod = resolution_log2 - res
        result = block(result, res)
        img = to_rgb(result)
        images_skip = layers.upscale()(images_skip)
        layer_lod = lod_in - lod
        output = img + (images_skip - img) * tf.clip_by_value(layer_lod, 0.0, 1.0)

    return tf.keras.models.Model(inputs=[latents_input, lod_input], outputs=output)


# model = generator_model()
# tf.keras.utils.plot_model(model, to_file='models/generator_model.png', show_shapes=True, show_layer_names=True, dpi=150)