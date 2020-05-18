import tensorflow as tf
import config


def get_ffhq():
    return tf.data.Dataset.list_files('E:/ffhq_256' + '/*.png').map(get_image).shuffle(
        config.buffer_size).batch(
        config.batch_size)


def get_mnist():
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('uint8')
    train_images = (train_images - 127.5) / 127.5

    return tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(
        config.buffer_size).batch(
        config.batch_size)


def get_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image
