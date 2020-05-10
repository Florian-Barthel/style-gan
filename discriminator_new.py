import tensorflow as tf
import layers
import numpy as np


class Discriminator(tf.keras.models.Model):
    def __init__(self, resolution=32, type=tf.dtypes.float32, num_channels=1, fmap_base=32):
        super(Discriminator, self).__init__()
        self.fmap_base = fmap_base
        self.type = type
        self.num_channels = num_channels
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2

        # Layers
        self.std_dev = layers.MinibatchStdev()
        self.from_rgb_first = layers.FromRGB(self.resolution_log2, self.fmap_base)
        self.blocks = dict()
        self.from_rgb_downscaled = dict()
        for res in range(self.resolution_log2, 2, -1):
            self.blocks[res] = layers.DiscBlock(res, self.fmap_base)
            self.from_rgb_downscaled[res] = layers.FromRGB(res - 1, self.fmap_base)
        self.last_block = layers.LastDiscBlock(self.fmap_base)

        # Functions
        self.downscale = layers.downscale()

    def call(self, inputs):
        image_input = inputs[0]
        lod_input = inputs[1]

        lod_remainder = lod_input - np.floor(lod_input)
        fist_layer = True

        x = self.from_rgb_first(image_input)
        for res in range(min(int(np.ceil(lod_input)) + 2, self.resolution_log2), 2, -1):
            if fist_layer and lod_remainder > 0:
                downscaled_image = self.downscale(image_input)
                y = self.from_rgb_downscaled[res](downscaled_image)
                x = self.blocks[res](x)
                x = x + (y - x) * lod_remainder
            else:
                x = self.blocks[res](x)
            fist_layer = False

        scores = self.last_block(x)
        return scores
