import tensorflow as tf

BUFFER_SIZE = 60000
# TODO: error with batchsize = 64
batch_size = 32
resolution = 32
latent_size = 32

epochs_per_lod = 2
lod_increase = 1 / epochs_per_lod

EPOCHS = 500
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, latent_size, 1])
