import matplotlib.pyplot as plt
import numpy as np
import config
import tensorflow as tf


def generate_and_save_images(generator_model, epoch, lod):
    images = generator_model([config.seed, lod])
    res = int(np.sqrt(config.num_examples_to_generate))
    resolution = int(2 ** (np.ceil(lod) + 2))
    figure_title = 'LoD: {}, Iteration: {}, Resolution: {}x{}'.format(lod,
                                                                      epoch * config.epoch_iterations * config.minibatch_repeat,
                                                                      resolution,
                                                                      resolution)
    fig, axs = plt.subplots(res, res, constrained_layout=True)
    counter = 0
    for i in range(res):
        for j in range(res):
            axs[i][j].imshow(tf.cast(tf.clip_by_value(images[counter, :, :, :] * 127.5 + 127.5, 0, 255), tf.uint8))
            axs[i][j].axis('off')
            counter += 1
    fig.suptitle(figure_title)
    plt.savefig(config.result_folder + '/image_at_iteration_{:04d}.png'.format(epoch), dpi=300)
    plt.show()
