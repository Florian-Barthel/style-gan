import tensorflow as tf

BUFFER_SIZE = 60000
# TODO: error with batchsize = 64
batch_size = 16
resolution = 32

EPOCHS = 500
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, resolution, 1])
