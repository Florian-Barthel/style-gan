import tensorflow as tf
import generator_new
import generator_loss
import discriminator_new
import discriminator_loss
import os
import config
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
import loss

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

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(config.BUFFER_SIZE).batch(config.batch_size)

generator_model = generator_new.Generator(
    num_mapping_layers=config.num_mapping_layers,
    mapping_fmaps=config.mapping_fmaps,
    resolution=config.resolution,
    fmap_base=config.fmap_base,
    num_channels=1)
discriminator_model = discriminator_new.Discriminator(
    resolution=config.resolution,
    fmap_base=config.fmap_base,
    num_channels=1)

generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()


@tf.function
def train(images, latents, lod):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images_out = generator_model([latents, lod])
        real_scores = discriminator_model([images, lod])
        fake_scores = discriminator_model([fake_images_out, lod])

        disc_loss = loss.wasserstein_loss(1, real_scores) + loss.wasserstein_loss(-1, fake_scores)
        gen_loss = loss.wasserstein_loss(1, fake_scores)

    gen_vars = generator_model.trainable_variables
    disc_vars = discriminator_model.trainable_variables

    gradients_of_generator = gen_tape.gradient(gen_loss, gen_vars)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc_vars)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_vars))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_vars))
    return gen_loss, disc_loss


def init(images_batch):
    lod = 0.0

    latents = tf.Variable(np.random.rand(config.batch_size, config.latent_size, 1) * 2 - 1,
                          dtype=tf.dtypes.float32,
                          trainable=False)

    while lod <= config.max_lod:
        lod_res = int(2 ** (np.ceil(lod) + 2))
        resized_batch = tf.image.resize(images_batch, [lod_res, lod_res],
                                        method=tf.image.ResizeMethod.AREA)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images_out = generator_model([latents, lod])
            real_scores = discriminator_model([resized_batch, lod])
            fake_scores = discriminator_model([fake_images_out, lod])

            disc_loss = loss.wasserstein_loss(1, real_scores) + loss.wasserstein_loss(-1, fake_scores)
            gen_loss = loss.wasserstein_loss(1, fake_scores)

        gen_vars = generator_model.trainable_variables
        disc_vars = discriminator_model.trainable_variables

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
    iteration = 0
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            lod_res = int(2 ** (np.floor(lod) + 2))
            resized_batch = tf.image.resize(image_batch, [lod_res, lod_res],
                                            method=tf.image.ResizeMethod.AREA)
            latents = tf.Variable(np.random.rand(config.batch_size, config.latent_size, 1) * 2 - 1,
                                  dtype=tf.dtypes.float32,
                                  trainable=False)
            gen_loss, disc_loss = train(resized_batch, latents, np.float32(lod))

            with summary_writer.as_default():
                tf.summary.scalar('gen_loss', gen_loss, step=generator_optimizer.iterations)
                tf.summary.scalar('disc_loss', disc_loss, step=discriminator_optimizer.iterations)
            iteration += 1

        # if epoch % 10:
        generate_and_save_images(epoch + 1, config.seed, lod)
        print('lod:', lod)
        print('gen_loss:', gen_loss)
        print('dis_loss:', disc_loss)

        if (epoch + 1) % config.epochs_per_lod == 0:
            increase_lod = not increase_lod

        if increase_lod and lod < config.max_lod:
            lod = np.around(lod + config.lod_increase, decimals=config.lod_decimals)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


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
train_loop(train_dataset, config.EPOCHS)
