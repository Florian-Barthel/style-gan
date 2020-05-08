import tensorflow as tf
import utils
import numpy as np
import layers


class Discriminator(tf.keras.models.Model):
    def __init__(self, resolution=32, type=tf.dtypes.float32, num_channels=1, fmap_base=32):
        super(Discriminator, self).__init__()
        self.fmap_base = fmap_base
        self.type = type
        self.num_channels = num_channels
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2

    def __call__(self, inputs):
        image_input = inputs[0]
        lod_input = inputs[1]

        # lod_counter = int(np.ceil(lod_input))
        lod_remainder = lod_input - int(lod_input)
        fist_layer = True

        x = self.from_rgb(image_input, self.resolution_log2)
        for res in range(min(int(np.ceil(lod_input)) + 2, self.resolution_log2), 2, -1):

            if fist_layer and lod_remainder > 0:
                downscaled_image = layers.downscale()(image_input)
                y = self.from_rgb(downscaled_image, res - 1)
                x = self.block(x, res)
                x = x + (y - x) * lod_remainder
            else:
                x = self.block(x, res)
            fist_layer = False

        x = layers.MinibatchStdev()(x)
        x = layers.conv2d(filters=self.num_filters(1), kernel_size=(3, 3))(x)
        x = layers.activation()(x)
        x = layers.dense(units=self.num_filters(0))(x)
        x = layers.activation()(x)
        x = layers.dense(units=1)(x)
        scores = layers.activation()(x)
        return scores

    def num_filters(self, stage):
        return int(self.fmap_base / (2.0 ** stage))

    def from_rgb(self, x, res):
        x = layers.conv2d(filters=self.num_filters(res - 1), kernel_size=(1, 1))(x)
        x = layers.activation()(x)
        return x

    def block(self, x, res):
        x = layers.conv2d(filters=self.num_filters(res - 1), kernel_size=(3, 3))(x)
        x = layers.activation()(x)
        x = layers.conv2d(filters=self.num_filters(res - 2), kernel_size=(3, 3))(x)
        x = layers.downscale()(x)
        return x
