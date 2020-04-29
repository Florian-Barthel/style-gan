import tensorflow as tf
import utils
import numpy as np
import layers


def discriminator_model(
        # images_in,  # First input: Images [minibatch, channel, height, width].
        num_channels=1,  # Number of input color channels. Overridden based on dataset.
        resolution=32,  # Input resolution. Overridden based on dataset.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        dtype='float32',  # Data type to use for activations and outputs.
        **_kwargs):  # Ignore unrecognized keyword args.

    model = tf.keras.models.Sequential()

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


    # images_in.set_shape([None, num_channels, resolution, resolution])
    # images_in = tf.cast(images_in, dtype)

    # Building blocks.
    def fromrgb(resolution):  # res = 2..resolution_log2
        fromrgb_model = tf.keras.models.Sequential()
        fromrgb_model.add(layers.conv2D(filters=nf(resolution - 1), kernel_size=(1, 1)))
        fromrgb_model.add(layers.apply_bias())
        fromrgb_model.add(layers.activation())
        return fromrgb_model

    def block(resolution):  # res = 2..resolution_log2
        block_model = tf.keras.models.Sequential()
        if resolution >= 3:  # 8x8 and up
            block_model.add(layers.conv2d(filters=nf(resolution - 1), kernel_size=(3, 3)))
            block_model.add(layers.apply_bias())
            block_model.add(layers.activation())

            block_model.add(layers.blur2d())
            block_model.add(layers.conv2d(filters=nf(resolution - 2), kernel_size=(3, 3)))
            block_model.add(layers.apply_bias())
            block_model.add(layers.downscale2d())
        else:  # 4x4
            if mbstd_group_size > 1:
                block_model.add(layers.minibatch_stddev())
            block_model.add(layers.conv2D(filters=nf(resolution - 1), kernel_size=(3, 3)))
            block_model.add(layers.apply_bias())
            block_model.add(layers.activation())

            block_model.add(layers.dense(units=nf(resolution - 2)))
            block_model.add(layers.apply_bias())
            block_model.add(layers.activation())

            block_model.add(layers.dense(units=1))
            block_model.add(layers.apply_bias())
            block_model.add(layers.activation())
        return block_model


    # linear structure
    model.add(fromrgb(resolution_log2))
    for res in range(resolution_log2, 2, -1):
        model.add(block(res))
    model.add(block(2))

    return model
