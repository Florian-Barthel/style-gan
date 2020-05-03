import tensorflow as tf

BUFFER_SIZE = 60000
batch_size = 32
resolution = 32

EPOCHS = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, resolution, 1])
