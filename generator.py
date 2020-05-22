import tensorflow as tf
from tensorflow.keras.models import Model
import layers
import numpy as np


class Generator(Model):
    def __init__(self,
                 num_mapping_layers=8,
                 mapping_fmaps=32,
                 resolution=64,
                 type=tf.float32,
                 num_channels=3,
                 fmap_base=32,
                 lr_mul=0.01,
                 gain=np.sqrt(2),
                 use_wscale=True):

        super(Generator, self).__init__()

        # Config vars
        self.resolution_log2 = int(np.log2(resolution))

        # Layers
        self.pixel_norm = layers.PixelNorm()
        self.mapping_layers = layers.Mapping(num_mapping_layers, mapping_fmaps, lr_mul=lr_mul, type=type, use_wscale=use_wscale)
        self.first_gen_block = layers.FirstGenBlock(fmap_base=fmap_base, type=type, gain=gain, use_wscale=use_wscale)
        self.blocks = dict()
        self.to_rgb_first = layers.ToRGB(num_channels, use_wscale=use_wscale)
        self.to_rgb_new = dict()
        self.to_rgb_old = dict()
        self.to_rgb_last = dict()
        self.to_rgb_last_mix = dict()
        for res in range(3, self.resolution_log2 + 1):
            self.to_rgb_old[res] = layers.ToRGB(num_channels, use_wscale=use_wscale)
            self.blocks[res] = layers.GenBlock(res=res, fmap_base=fmap_base, type=type, use_wscale=use_wscale, gain=gain)
            self.to_rgb_new[res] = layers.ToRGB(num_channels, use_wscale=use_wscale)
            self.to_rgb_last[res] = layers.ToRGB(num_channels, use_wscale=use_wscale)
            self.to_rgb_last_mix[res] = layers.ToRGB(num_channels, use_wscale=use_wscale)

        # Functions
        self.upscale = layers.upscale(2)

    def call(self, inputs, trainable=True, mask=None):
        latents_input = inputs[0]
        lod_input = inputs[1]
        latents = self.pixel_norm(latents_input)
        latents = self.mapping_layers(latents)

        x = self.first_gen_block(latents)
        result = self.to_rgb_first(x)
        lod_counter = int(np.ceil(lod_input))
        lod_remainder = lod_input - np.floor(lod_input)

        for res in range(3, min(int(np.ceil(lod_input)) + 3, self.resolution_log2 + 1)):
            if lod_counter == 1 and lod_remainder > 0:
                rgb_image = self.to_rgb_old[res](x)
                prev = self.upscale(rgb_image)
                x = self.blocks[res]([x, latents])
                new = self.to_rgb_new[res](x)
                x = new + (prev - new) * (1 - lod_remainder)
                result = self.to_rgb_last_mix[res](x)
            else:
                x = self.blocks[res]([x, latents])
                result = self.to_rgb_last[res](x)
            lod_counter -= 1
        return result
