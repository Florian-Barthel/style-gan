import numpy as np
import datetime
import tensorflow as tf

batch_size = 16
resolution = 128
latent_size = 512
num_mapping_layers = 8
mapping_fmaps = 512
fmap_base = 512
fmap_max = 512
num_channels = 3

iterations_per_lod = 40
lod_decimals = 3
epoch_iterations = 250
lod_increase = 1 / iterations_per_lod
max_lod = int(np.log2(resolution)) - 2

epochs = 10000
minibatch_repeat = 4
num_examples_to_generate = 16


# design options
reset_optimizer = True
flip_images = True
use_wscale = True

# Folders
log_dir = "logs/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_folder = 'runs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

seed = tf.random.normal([batch_size, latent_size, 1], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
