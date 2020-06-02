import tensorflow as tf
import layers
import numpy as np


class Discriminator(tf.keras.models.Model):
    def __init__(self,
                 resolution,
                 fmap_base,
                 use_wscale,
                 gain=np.sqrt(2)):

        super(Discriminator, self).__init__()

        # Config vars
        self.resolution_log2 = int(np.log2(resolution))

        # Layers
        self.from_rgb = dict()
        self.blocks = dict()
        for res in range(self.resolution_log2, 2, -1):
            self.from_rgb[res] = layers.FromRGB(res, fmap_base, use_wscale=use_wscale, gain=gain)
            self.blocks[res] = layers.DiscBlock(res, fmap_base, use_wscale=use_wscale, gain=gain)
        self.from_rgb[2] = layers.FromRGB(2, fmap_base, use_wscale=use_wscale, gain=gain)
        self.last_block = layers.LastDiscBlock(fmap_base, gain=gain, use_wscale=use_wscale)

        # Functions
        self.downscale = layers.downscale()

    def call(self, inputs, trainable=True, mask=None):
        image_input = inputs[0]
        lod_input = inputs[1]

        lod_remainder = lod_input - np.floor(lod_input)
        fist_layer = True

        lod_res = int(np.ceil(lod_input)) + 2
        x = self.from_rgb[lod_res](image_input)
        for res in range(min(lod_res, self.resolution_log2), 2, -1):
            x = self.blocks[res](x)
            if fist_layer and lod_remainder > 0:
                downscaled_image = self.downscale(image_input)
                y = self.from_rgb[res - 1](downscaled_image)
                x = x + (y - x) * (1 - lod_remainder)
            fist_layer = False

        scores = self.last_block(x)
        return scores
