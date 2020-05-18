import numpy as np

buffer_size = 1000
batch_size = 16
resolution = 256
latent_size = 128
num_mapping_layers = 8
mapping_fmaps = 128
fmap_base = 128
fmap_max = 128
num_channels = 3

epochs_per_lod = 20
lod_decimals = 2
lod_iterations = 1000

epochs = 500
num_examples_to_generate = 16
seed = np.random.rand(batch_size, latent_size, 1)

lod_increase = 1 / epochs_per_lod
max_lod = int(np.log2(resolution)) - 2


# design options
reset_optimizer = True
