import tensorflow as tf
import layers

factor = 2

x = tf.zeros([1, 4, 4, 512])
s = tf.shape(x)

reshaped1 = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
tiled = tf.tile(reshaped1, [1, factor, factor, 1, 1, 1])
reshaped2 = tf.reshape(tiled, [-1, s[1] * factor, s[2] * factor, s[3]])

print(x)
