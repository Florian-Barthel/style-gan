import tensorflow as tf

BUFFER_SIZE = 60000
batch_size = 8
resolution = 16

EPOCHS = 50
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, resolution, 1])
