import tensorflow as tf
from tensorflow.keras.models import Model
import layers
import numpy as np
import config


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
                 use_wscale=True,
                 dlatent_avg_beta=0.995,
                 truncation_psi=0.7,
                 truncation_cutoff=8,
                 style_mixing_prob=0.9):

        super(Generator, self).__init__()

        # Config vars
        resolution_log2 = int(np.log2(resolution))
        self.num_style_layers = (resolution_log2 - 1) * 2
        self.style_mixing_prob = style_mixing_prob

        # Sub Networks
        self.mapping_network = MappingNetwork(num_mapping_layers=num_mapping_layers,
                                              mapping_fmaps=mapping_fmaps,
                                              lr_mul=lr_mul,
                                              use_wscale=use_wscale,
                                              type=type,
                                              num_style_layers=self.num_style_layers)
        self.synthesis_network = SynthesisNetwork(use_wscale=use_wscale,
                                                  type=type,
                                                  num_channels=num_channels,
                                                  fmap_base=fmap_base,
                                                  gain=gain,
                                                  resolution_log2=resolution_log2)

        # Layers
        self.get_dlatent_avg = layers.GetDlatentAvg(dlatent_avg_beta)
        self.apply_truncation = layers.ApplyTruncation(self.num_style_layers, truncation_cutoff, truncation_psi)

    def call(self, inputs, trainable=True, mask=None):
        latents_input = inputs[0]
        lod_input = inputs[1]
        is_validation = inputs[2]
        dlatents = self.mapping_network(inputs=[latents_input], trainable=trainable)
        dlatents, dlatent_avg = self.get_dlatent_avg([dlatents, is_validation])

        if config.use_style_mix:
            latents2 = tf.random.normal(tf.shape(latents_input))
            dlatents2 = self.mapping_network(inputs=[latents2], trainable=trainable)
            layer_idx = np.arange(self.num_style_layers)[np.newaxis, :, np.newaxis]
            cur_layers = tf.constant(2 * np.ceil(lod_input) + 2, dtype=tf.int32)
            mixing_cutoff = tf.cond(
                tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob,
                lambda: tf.random.uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

        if is_validation == 1.0 and config.use_truncation:
            dlatents = self.apply_truncation([dlatents, dlatent_avg])
        return self.synthesis_network([dlatents, lod_input], trainable=trainable)


class MappingNetwork(Model):
    def __init__(self,
                 num_mapping_layers,
                 num_style_layers,
                 mapping_fmaps,
                 use_wscale,
                 lr_mul,
                 type):
        super(MappingNetwork, self).__init__()
        self.pixel_norm = layers.PixelNorm()
        self.mapping_layers = layers.Mapping(num_mapping_layers, mapping_fmaps, num_style_layers=num_style_layers,
                                             lr_mul=lr_mul, type=type, use_wscale=use_wscale)

    def call(self, inputs, trainable=True, mask=None):
        latents_input = inputs[0]
        latents = self.pixel_norm(latents_input)
        dlatents = self.mapping_layers(latents)
        return dlatents


class SynthesisNetwork(Model):
    def __init__(self,
                 use_wscale,
                 type,
                 num_channels,
                 fmap_base,
                 gain,
                 resolution_log2):
        super(SynthesisNetwork, self).__init__()

        # Config vars
        self.resolution_log2 = resolution_log2

        # Layers
        self.first_gen_block = layers.FirstGenBlock(fmap_base=fmap_base, type=type, gain=gain, use_wscale=use_wscale)
        self.blocks = dict()
        self.to_rgb = dict()
        self.to_rgb[2] = layers.ToRGB(num_channels, use_wscale=use_wscale)
        for res in range(3, resolution_log2 + 1):
            self.blocks[res] = layers.GenBlock(res=res, fmap_base=fmap_base, type=type, use_wscale=use_wscale, gain=gain)
            self.to_rgb[res] = layers.ToRGB(num_channels, use_wscale=use_wscale)

        # Functions
        self.upscale = layers.upscale(2)

    def call(self, inputs, trainable=True, mask=None):
        dlatents_input = inputs[0]
        lod_input = inputs[1]
        x = self.first_gen_block(dlatents_input)
        if lod_input == 0.0:
            return self.to_rgb[2](x)

        lod_counter = int(np.ceil(lod_input))
        lod_remainder = lod_input - np.floor(lod_input)

        for res in range(3, min(int(np.ceil(lod_input)) + 3, self.resolution_log2 + 1)):
            if lod_counter == 1:
                if lod_remainder > 0:
                    rgb_image = self.to_rgb[res - 1](x)
                    prev = self.upscale(rgb_image)
                    x = self.blocks[res]([x, dlatents_input])
                    new = self.to_rgb[res](x)
                    return new + (prev - new) * (1 - lod_remainder)
                else:
                    x = self.blocks[res]([x, dlatents_input])
                    return self.to_rgb[res](x)
            else:
                x = self.blocks[res]([x, dlatents_input])
                lod_counter -= 1
