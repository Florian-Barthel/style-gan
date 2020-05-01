import tensorflow as tf
import numpy as np


BUFFER_SIZE = 60000
BATCH_SIZE = 16

EPOCHS = 50
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, 32, 1])


gain = np.sqrt(2)