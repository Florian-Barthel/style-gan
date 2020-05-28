import matplotlib.pyplot as plt
import numpy as np
import os
import config
import tensorflow as tf


def generate_and_save_images(generator_model, num_images, lod, iteration):
    images = generator_model([config.seed, lod, np.float32(1)], trainable=False)
    res = int(np.sqrt(config.num_examples_to_generate))
    resolution = int(2 ** (np.ceil(lod) + 2))
    figure_title = 'LoD: {:.3f}  |  num_images: {}  |  resolution: {}x{}'.format(lod, num_images, resolution, resolution)
    fig, axs = plt.subplots(res, res)
    counter = 0
    for i in range(res):
        for j in range(res):
            axs[i][j].imshow(tf.cast(tf.clip_by_value(images[counter, :, :, :] * 127.5 + 127.5, 0, 255), tf.uint8))
            axs[i][j].axis('off')
            counter += 1
    fig.suptitle(figure_title)
    image_folder = config.result_folder + '/images/'
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    plt.savefig(image_folder + '/image_at_iteration_{:04d}.png'.format(iteration), dpi=300)
    plt.close('all')