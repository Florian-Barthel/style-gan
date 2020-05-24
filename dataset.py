import tensorflow as tf
import numpy as np
import config


def get_ffhq(res):
    return tf.data.Dataset.list_files('E:/ffhq_' + str(res) + '/*.png').map(
        get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
        config.batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()


def get_mnist():
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('uint8')
    train_images = (train_images - 127.5) / 127.5

    return tf.data.Dataset.from_tensor_slices(
        train_images).batch(
        config.batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)


def get_latent():
    return tf.data.Dataset.range(config.batch_size * 2).map(
        create_random_vector, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
        config.batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)


@tf.function
def get_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_png(image, channels=3)
    if np.random.normal() > 0 and config.flip_images:
        image = tf.image.flip_left_right(image)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image


def create_random_vector(x):
    return tf.random.normal([config.latent_size, 1], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
