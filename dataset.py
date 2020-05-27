import tensorflow as tf
import numpy as np
import config


def get_ffhq(res, batch_size):
    return tf.data.Dataset.list_files('E:/ffhq_' + str(res) + '/*.png').map(
        get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat().batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def get_mnist(batch_size):
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('uint8')
    train_images = (train_images - 127.5) / 127.5

    return tf.data.Dataset.from_tensor_slices(
        train_images).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)


@tf.function
def get_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_png(image, channels=3)
    if np.random.normal() > 0 and config.flip_images:
        image = tf.image.flip_left_right(image)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image
