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
        # self.latest_input_shape = None

        # Config vars
        self.resolution_log2 = int(np.log2(resolution))

        # Layers
        self.pixel_norm = layers.PixelNorm()
        self.mapping_layers = layers.Mapping(num_mapping_layers, mapping_fmaps, lr_mul=lr_mul, type=type, use_wscale=use_wscale)
        self.first_gen_block = layers.FirstGenBlock(fmap_base=fmap_base, type=type, gain=gain, use_wscale=use_wscale)
        self.blocks = dict()
        self.to_rgb = dict()
        self.to_rgb[2] = layers.ToRGB(num_channels, use_wscale=use_wscale)
        for res in range(3, self.resolution_log2 + 1):
            self.blocks[res] = layers.GenBlock(res=res, fmap_base=fmap_base, type=type, use_wscale=use_wscale, gain=gain)
            self.to_rgb[res] = layers.ToRGB(num_channels, use_wscale=use_wscale)

        # Functions
        self.upscale = layers.upscale(2)

    def call(self, inputs, trainable=True, mask=None):
        # self.latest_input_shape = [tf.shape(inputs[0]), tf.shape(inputs[1])]
        latents_input = inputs[0]
        lod_input = inputs[1]
        latents = self.pixel_norm(latents_input)
        latents = self.mapping_layers(latents)

        x = self.first_gen_block(latents)
        if lod_input == 0.0:
            return self.to_rgb[2](x)

        lod_counter = int(np.ceil(lod_input))
        lod_remainder = lod_input - np.floor(lod_input)

        for res in range(3, min(int(np.ceil(lod_input)) + 3, self.resolution_log2 + 1)):
            if lod_counter == 1:
                if lod_remainder > 0:
                    rgb_image = self.to_rgb[res - 1](x)
                    prev = self.upscale(rgb_image)
                    x = self.blocks[res]([x, latents])
                    new = self.to_rgb[res](x)
                    return new + (prev - new) * (1 - lod_remainder)
                else:
                    x = self.blocks[res]([x, latents])
                    return self.to_rgb[res](x)
            else:
                x = self.blocks[res]([x, latents])
                lod_counter -= 1

    # def save_model(self, path):
    #     self._set_inputs([tf.random.normal(self.latest_input_shape[0]), tf.random.normal(self.latest_input_shape[1])])
    #     self.save(filepath=path)
