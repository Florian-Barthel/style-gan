import numpy as np
import datetime

batch_size = 16
resolution = 256
latent_size = 512
num_mapping_layers = 8
mapping_fmaps = 512
fmap_base = 256
fmap_max = 256
num_channels = 3

iterations_per_lod = 20
lod_decimals = 3
lod_iterations = 2000
lod_increase = 1 / iterations_per_lod
max_lod = int(np.log2(resolution)) - 2

epochs = 10000
num_examples_to_generate = 16


# design options
reset_optimizer = True
flip_images = True

# Folders
log_dir = "logs/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_folder = 'runs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")