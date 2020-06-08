import datetime
import tensorflow as tf
import numpy as np

# Network
resolution = 128
latent_size = 512
dlatent_size = 512
num_mapping_layers = 8
mapping_fmaps = 512
fmap_base = 4096 * 2
fmap_max = 512
num_channels = 3

# Training
iterations_per_lod_dict = {4: 5,  8: 10, 16: 20, 32: 40, 64: 80, 128: 80}
minibatch_dict =          {4: 64, 8: 32, 16: 16, 32:  8, 64:  4, 128:  4}
# minibatch_dict =        {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16}
minibatch_repeat = 4
epoch_iterations = 500
epochs = 10000
evaluation_interval = 10
save_images_interval = 2
fid_num_images = 10000
initial_lod = 1.0
initial_res = 2 ** (int(initial_lod) + 2)

# Design Options
reset_optimizer = True
flip_images = True
use_wscale = True
use_truncation = True
use_style_mix = True

# Output
log_dir = "./logs/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_folder = './runs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
num_examples_to_generate = 16
min_batch_size = minibatch_dict[min(minibatch_dict.keys(), key=(lambda k: minibatch_dict[k]))]
num_batches = int(np.ceil(num_examples_to_generate / min_batch_size))
seed = tf.random.normal([num_batches, min_batch_size, latent_size], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
