import tensorflow as tf
import numpy as np
import layers


class Generator(tf.keras.models.Model):
    def __init__(self, num_mapping_layers=8, mapping_fmaps=32, resolution=64, type=tf.dtypes.float32, num_channels=1,
                 fmap_base=32):
        super(Generator, self).__init__()
        self.num_mapping_layers = num_mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.fmap_base = fmap_base
        self.type = type
        self.num_channels = num_channels
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.noise_inputs = []
        self.x = None

    def __call__(self, inputs):
        latents_input = inputs[0]
        lod_input = inputs[1]
        latents = layers.PixelNorm()(latents_input)

        # Mapping Network
        for layer_idx in range(self.num_mapping_layers):
            latents = layers.dense(self.mapping_fmaps)(latents)
            latents = layers.activation()(latents)

        # Generate Noise
        for layer_idx in range(self.num_layers):
            noise_res = layer_idx // 2 + 2
            shape = [1, 2 ** noise_res, 2 ** noise_res, 1]
            self.noise_inputs.append(tf.Variable(tf.random.normal(shape), trainable=False, dtype=self.type))

        # first layer 4 x 4
        batchsize = tf.shape(latents)[0]
        x = tf.Variable(tf.random.normal([1, 4, 4, self.num_filters(1)]), trainable=True, dtype=self.type,
                        name='CONSTANT')
        x = layers.Tile([batchsize, 1, 1, 1])(x)
        x = self.layer_epilogue(x, latents, 0)
        x = layers.conv2d(self.num_filters(1), (3, 3))(x)
        x = self.layer_epilogue(x, latents, 1)

        lod_counter = int(np.ceil(lod_input)) - 1
        lod_remainder = lod_input - int(lod_input)
        # Remaining Blocks
        if int(np.ceil(lod_input)) > self.resolution_log2:
            print('WARNING: LoD = {}, while log(resolution) = {}'.format(lod_input, self.resolution_log2))

        for res in range(3, min(int(np.ceil(lod_input)) + 3, self.resolution_log2 + 1)):
            if lod_counter == 1 and lod_remainder > 0:
                rgb_image = self.to_rgb(x)
                prev = layers.upscale()(rgb_image)
                x = self.block(x, latents, res)
                new = self.to_rgb(x)
                return new + (prev - new) * (1 - lod_remainder)

            x = self.block(x, latents, res)
            lod_counter -= 1

        img = self.to_rgb(x)
        return img

    def block(self, x, latents, res):
        x = layers.upscale()(x)
        x = layers.conv2d(self.num_filters(res - 1), (3, 3))(x)
        # TODO: create layers for blur
        x = self.layer_epilogue(x, latents, res * 2 - 4)
        x = layers.conv2d(self.num_filters(res - 1), (3, 3))(x)
        x = self.layer_epilogue(x, latents, res * 2 - 3)
        return x

    def layer_epilogue(self, x, latents, layer_index):
        x = layers.ApplyNoiseWithWeights(noise=self.noise_inputs[layer_index])(x)
        x = layers.activation()(x)
        x = layers.InstanceNorm()(x)
        style = layers.IndexSlice(layer_index)(latents)
        style = layers.dense(x.shape[3] * 2)(style)
        x = layers.StyleMod2()([x, style])
        return x

    def to_rgb(self, x):
        return layers.conv2d(self.num_channels, (1, 1))(x)

    def num_filters(self, stage):
        return int(self.fmap_base / (2.0 ** stage))

