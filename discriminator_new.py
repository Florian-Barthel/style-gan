import tensorflow as tf
import layers
import numpy as np


class Discriminator(tf.keras.models.Model):
    def __init__(self, resolution=32, type=tf.dtypes.float32, num_channels=1, fmap_base=32):
        super(Discriminator, self).__init__()
        self.fmap_base = fmap_base
        self.type = type
        self.num_channels = num_channels
        self.resolution_tensor = tf.Variable(resolution, dtype=tf.dtypes.float32, trainable=False)
        self.resolution_log2 = tf.cast(
            tf.math.log(self.resolution_tensor) / tf.math.log(
                tf.Variable(2, dtype=tf.dtypes.float32, trainable=False)),
            dtype=tf.dtypes.int32)
        self.zero = tf.Variable(0, dtype=type, trainable=False)
        self.counter =tf.Variable(0, dtype=tf.dtypes.int32, trainable=False)

        # Layers
        self.std_dev = layers.MinibatchStdev()
        self.from_rgb_first = dict()
        self.blocks = dict()
        self.from_rgb_downscaled = dict()
        for res in range(self.resolution_log2, 2, -1):
            self.from_rgb_first[res] = layers.FromRGB(res, self.fmap_base)
            self.blocks[res] = layers.DiscBlock(res, self.fmap_base)
            self.from_rgb_downscaled[res] = layers.FromRGB(res - 1, self.fmap_base)
        self.from_rgb_first[2] = layers.FromRGB(2, self.fmap_base)
        self.last_block = layers.LastDiscBlock(self.fmap_base)


        # Functions
        self.downscale = layers.downscale()

    @tf.function
    def __call__(self, inputs):
        image_input = inputs[0]
        lod_input = inputs[1]

        lod_remainder = lod_input - tf.math.floor(lod_input)
        fist_layer = True

        lod_res = tf.cast(tf.math.ceil(lod_input), dtype=tf.dtypes.int32) + 2

        # Quick fix for: "TypeError: Tensor is unhashable if Tensor equality is enabled"
        for res in range(self.resolution_log2 + 1):
            if tf.equal(lod_res, res):
                x = self.from_rgb_first[res](image_input)

        for res in range(min(lod_res, self.resolution_log2), 2, -1):
            x = self.blocks[res](x)
            if fist_layer and tf.math.greater(lod_remainder, 0):
                downscaled_image = self.downscale(image_input)
                y = self.from_rgb_downscaled[res](downscaled_image)
                x = x + (y - x) * lod_remainder
            fist_layer = False

        scores = self.last_block(x)
        return scores
