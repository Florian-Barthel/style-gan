import numpy as np
import datetime
import tensorflow as tf

# batch_size = 16
resolution = 128
latent_size = 512
num_mapping_layers = 8
mapping_fmaps = 512
fmap_base = 4096
fmap_max = 512
num_channels = 3

iterations_per_lod = 40
iterations_per_lod_dict = {4: 40, 8: 40, 16: 40, 32: 80, 64: 160, 128: 160}
minibatch_dict = {4: 16, 8: 16, 16: 16, 32: 8, 64: 4, 128: 4}
minibatch_repeat = 4
epoch_iterations = 250
lod_increase = 1 / iterations_per_lod
max_lod = int(np.log2(resolution)) - 2

epochs = 10000
evaluation_interval = 10
fid_num_images = 10000
num_examples_to_generate = 16

# design options
reset_optimizer = True
flip_images = True
use_wscale = True

# Folders
log_dir = "./logs/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_folder = './runs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

seed = tf.random.normal([num_examples_to_generate, latent_size, 1], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
