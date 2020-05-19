import numpy as np

buffer_size = 1000
batch_size = 16
resolution = 256
latent_size = 256
num_mapping_layers = 8
mapping_fmaps = 256
fmap_base = 256
fmap_max = 256
num_channels = 3

iterations_per_lod = 10
lod_decimals = 3
lod_iterations = 1000
lod_increase = 1 / iterations_per_lod
max_lod = int(np.log2(resolution)) - 2

epochs = 10000
num_examples_to_generate = 16
seed = np.random.rand(batch_size, latent_size, 1)

# design options
reset_optimizer = True
