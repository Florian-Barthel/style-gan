import tensorflow as tf
import numpy as np
import layers

def discriminator_model(
        resolution=32,
        dtype='float32',
        fmap_base=32,
        number_of_channels=1):

    resolution_log2 = int(np.log2(resolution))
    input_image = tf.keras.Input(shape=[None, None, number_of_channels], dtype=dtype, name='images')
    lod_input = tf.keras.Input(shape=[1], dtype=dtype, name='lod')

    # quick fix to remove the batch dimension
    lod_in = tf.reduce_max(lod_input)

    def num_filters(stage):
        return int(fmap_base / (2.0 ** stage))

    def from_RGB(x, res):
        conv = layers.conv2d(filters=num_filters(res - 1), kernel_size=(1, 1))(x)
        act = layers.activation()(conv)
        return act

    def block(x, res):
        conv1 = layers.conv2d(filters=num_filters(res - 1), kernel_size=(3, 3))(x)
        act = layers.activation()(conv1)
        conv2 = layers.conv2d(filters=num_filters(res - 2), kernel_size=(3, 3))(act)
        block_result = layers.downscale()(conv2)
        return block_result

    # img = input_image
    def create_graph():
        def graph(img):
            x = from_RGB(img, resolution_log2)
            for block_res in range(resolution_log2, 2, -1):
                lod = resolution_log2 - block_res
                if lod >= 0:
                    x = block(x, block_res)
                    img = layers.downscale()(img)
                    y = from_RGB(img, block_res - 1)
                    layer_lod = lod_in - lod
                    x = x + (y - x) * tf.clip_by_value(layer_lod, 0.0, 1.0)
            return x
        return graph

    # converted_graph = tf.autograph.to_graph(graph)
    result = create_graph()(input_image)
    # final layers
    std_dev = layers.MinibatchStdev()(result)
    conv1 = layers.conv2d(filters=num_filters(1), kernel_size=(3, 3))(std_dev)
    act1 = layers.activation()(conv1)
    dense1 = layers.dense(units=num_filters(0))(act1)
    act2 = layers.activation()(dense1)
    dense2 = layers.dense(units=1)(act2)
    scores_out = layers.activation()(dense2)

    return tf.keras.models.Model(inputs=[input_image, lod_input], outputs=scores_out)


model = discriminator_model()
tf.keras.utils.plot_model(model, to_file='models/discriminator_model.png', show_shapes=True, show_layer_names=True, dpi=150)
