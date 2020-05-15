import numpy as np

buffer_size = 60000
batch_size = 32
resolution = 32
latent_size = 128
num_mapping_layers = 8
mapping_fmaps = 128
fmap_base = 128

epochs_per_lod = 10
lod_decimals = 2

epochs = 500
num_examples_to_generate = 16
seed = np.random.rand(batch_size, latent_size, 1)

lod_increase = 1 / epochs_per_lod
max_lod = int(np.log2(resolution)) - 2


# design options
reset_optimizer = True
