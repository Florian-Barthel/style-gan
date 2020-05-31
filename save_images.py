import matplotlib.pyplot as plt
import numpy as np
import config
import tensorflow as tf
import time
import os

start = time.time()


def generate_and_save_images(generator_model, num_images, lod, iteration):

    image_res = int(2 ** (np.ceil(lod) + 2))
    images = np.empty([config.min_batch_size * config.num_batches, image_res, image_res, config.num_channels], dtype=np.float32)
    for i in range(config.num_batches):
        begin = i * config.min_batch_size
        end = (i + 1) * config.min_batch_size
        images[begin:end] = generator_model([config.seed[i, :, :], lod, np.float32(1)], trainable=False)

    hours = int((time.time() - start) / (60 * 60))
    figure_title = 'LoD: {:.3f}  |  num_images: {}  |  resolution: {}x{}  | time: {}h'.format(lod,
                                                                                              num_images,
                                                                                              image_res,
                                                                                              image_res,
                                                                                              hours)
    plot_res = int(np.sqrt(config.num_examples_to_generate))
    fig, axs = plt.subplots(plot_res, plot_res)
    counter = 0
    for i in range(plot_res):
        for j in range(plot_res):
            axs[i][j].imshow(tf.cast(tf.clip_by_value(images[counter, :, :, :] * 127.5 + 127.5, 0, 255), tf.uint8))
            axs[i][j].axis('off')
            counter += 1
    fig.suptitle(figure_title)
    image_folder = config.result_folder + '/images/'
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    plt.savefig(image_folder + '/image_at_iteration_{:04d}.png'.format(iteration), dpi=300)
    plt.close('all')
