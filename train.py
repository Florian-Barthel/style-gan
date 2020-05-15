import tensorflow as tf
import generator
import discriminator
import os
import config
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
import loss
import process_labels

# Set True for debugging
tf.config.experimental_run_functions_eagerly(False)

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

log_dir = "logs/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
summary_writer = tf.summary.create_file_writer(logdir=log_dir)

(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('uint8')
train_images = (train_images - 127.5) / 127.5

train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(
    config.buffer_size).batch(
    config.batch_size)

generator_model = generator.Generator(num_mapping_layers=config.num_mapping_layers,
                                      mapping_fmaps=config.mapping_fmaps,
                                      resolution=config.resolution,
                                      fmap_base=config.fmap_base,
                                      num_channels=1)

discriminator_model = discriminator.Discriminator(resolution=config.resolution,
                                                  fmap_base=config.fmap_base,
                                                  num_channels=1)

generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

gen_var_list = []
disc_var_list = []
var_list_index = 0


@tf.function
def train(images, latents, lod):
    global var_list_index
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images_out = generator_model([latents, lod])
        real_scores = discriminator_model([images, lod])
        fake_scores = discriminator_model([fake_images_out, lod])

        # disc_loss = loss.wasserstein_loss(1, real_scores) + loss.wasserstein_loss(-1, fake_scores)
        # gen_loss = loss.wasserstein_loss(1, fake_scores)
        disc_loss = loss.d_logistic(real_scores, fake_scores)
        gen_loss = loss.g_logistic(fake_scores)

    gen_vars = gen_var_list[var_list_index]
    disc_vars = disc_var_list[var_list_index]

    gradients_of_generator = gen_tape.gradient(gen_loss, gen_vars)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc_vars)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_vars))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_vars))
    return gen_loss, disc_loss


def init(image_batch):
    lod = 0.0

    while lod <= config.max_lod:
        lod_res = int(2 ** (np.ceil(lod) + 2))
        latents = tf.Variable(np.random.rand(config.batch_size, config.latent_size, 1) * 2 - 1,
                              dtype=tf.dtypes.float32,
                              trainable=False)
        resized_batch = process_labels.resize(image_batch, lod)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images_out = generator_model([latents, lod])
            real_scores = discriminator_model([resized_batch, lod])
            fake_scores = discriminator_model([fake_images_out, lod])

            # disc_loss = loss.wasserstein_loss(1, real_scores) + loss.wasserstein_loss(-1, fake_scores)
            # gen_loss = loss.wasserstein_loss(1, fake_scores)
            disc_loss = loss.d_logistic(real_scores, fake_scores)
            gen_loss = loss.g_logistic(fake_scores)

        gen_vars = generator_model.trainable_variables
        disc_vars = discriminator_model.trainable_variables

        gen_var_list.append(gen_vars)
        disc_var_list.append(disc_vars)

        gradients_of_generator = gen_tape.gradient(gen_loss, gen_vars)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, disc_vars)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_vars))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_vars))
        print('init layer:', lod_res)
        print('lod:', lod)
        print('num vars gen:', len(gen_vars))
        print('num vars disc:', len(disc_vars))
        lod += 0.5


def train_loop(dataset, epochs):
    lod = 0.0
    gen_loss = 0
    disc_loss = 0
    increase_lod = False
    iteration = 1
    global var_list_index
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            resized_batch = process_labels.resize(image_batch, lod)
            latents = tf.Variable(np.random.rand(config.batch_size, config.latent_size, 1) * 2 - 1,
                                  dtype=tf.dtypes.float32,
                                  trainable=False)
            gen_loss, disc_loss = train(resized_batch, latents, np.float32(lod))
            if iteration % 100:
                with summary_writer.as_default():
                    tf.summary.scalar('gen_loss', gen_loss, step=iteration)
                    tf.summary.scalar('disc_loss', disc_loss, step=iteration)
            iteration += 1

        generate_and_save_images(epoch + 1, config.seed, lod)
        print('lod:', lod)
        print('gen_loss:', gen_loss.numpy())
        print('dis_loss:', disc_loss.numpy())

        if (epoch + 1) % config.epochs_per_lod == 0:
            if config.reset_optimizer and not increase_lod:
                print('reset optimizer')
                for var in generator_optimizer.variables():
                    var.assign(tf.zeros_like(var))
                for var in discriminator_optimizer.variables():
                    var.assign(tf.zeros_like(var))
            increase_lod = not increase_lod
            var_list_index += 1

        if increase_lod and lod < config.max_lod:
            lod = np.around(lod + config.lod_increase, decimals=config.lod_decimals)

        print('Time for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


def generate_and_save_images(epoch, test_input, lod):
    images = generator_model([test_input, lod])

    plt.figure(figsize=(4, 4))
    for i in range(config.num_examples_to_generate):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


init(next(iter(train_dataset)))
train_loop(train_dataset, config.epochs)
