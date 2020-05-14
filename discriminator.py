import tensorflow as tf
import layers
import numpy as np


class Discriminator(tf.keras.models.Model):
    def __init__(self,
                 resolution=32,
                 type=tf.float32,
                 num_channels=1,
                 fmap_base=32):

        super(Discriminator, self).__init__()

        # Config vars
        self.fmap_base = fmap_base
        self.type = type
        self.num_channels = num_channels
        self.resolution_log2 = int(np.log2(resolution))

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

    def call(self, inputs, trainable=True, mask=None):
        image_input = inputs[0]
        lod_input = inputs[1]

        lod_remainder = lod_input - np.floor(lod_input)
        fist_layer = True

        lod_res = int(np.ceil(lod_input)) + 2
        x = self.from_rgb_first[lod_res](image_input)
        for res in range(min(lod_res, self.resolution_log2), 2, -1):
            x = self.blocks[res](x)
            if fist_layer and lod_remainder > 0:
                downscaled_image = self.downscale(image_input)
                y = self.from_rgb_downscaled[res](downscaled_image)
                x = x + (y - x) * (1 - lod_remainder)
            fist_layer = False

        scores = self.last_block(x)
        return scores
