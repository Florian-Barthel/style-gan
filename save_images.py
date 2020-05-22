import matplotlib.pyplot as plt
import numpy as np
import config
import tensorflow as tf


def generate_and_save_images(generator_model, epoch, lod):
    latent = np.random.rand(config.batch_size, config.latent_size, 1)
    images = generator_model([latent, lod])
    res = int(np.sqrt(config.num_examples_to_generate))
    plt.figure(figsize=(res, res))
    for i in range(config.num_examples_to_generate):
        plt.subplot(4, 4, i + 1)
        plt.imshow(tf.cast(tf.clip_by_value(images[i, :, :, :] * 127.5 + 127.5, 0, 255), tf.uint8))
        plt.axis('off')
    plt.savefig(config.result_folder + '/image_at_iteration_{:04d}.png'.format(epoch))
    plt.show()