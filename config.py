import tensorflow as tf
import numpy as np

BUFFER_SIZE = 60000
# TODO: error with batchsize = 64
batch_size = 32
resolution = 32
latent_size = 32
num_mapping_layers = 32
mapping_fmaps = 32
fmap_base = 32

epochs_per_lod = 2
lod_increase = np.float32(1 / epochs_per_lod)
max_lod = int(np.log2(resolution)) - 1

EPOCHS = 500
num_examples_to_generate = 16
seed = np.random.rand(batch_size, latent_size, 1)
