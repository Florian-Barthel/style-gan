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
# train_images = tf.image.resize(train_images, (config.resolution, config.resolution))

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

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator_model,
                                 discriminator=discriminator_model)


@tf.function
def train_both(images, lod):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        latents = np.random.rand(config.batch_size, config.latent_size, 1) * 2 - 1
        fake_images_out = generator_model([latents, lod])
        real_scores = discriminator_model([images, lod])
        fake_scores = discriminator_model([fake_images_out, lod])

        Disc_loss = loss.wasserstein_loss(1, real_scores) + loss.wasserstein_loss(-1, fake_scores)
        Gen_loss = loss.wasserstein_loss(1, fake_scores)

        acc_real = tf.reduce_sum(real_scores / 2 + 0.5) / config.batch_size
        acc_fake = (config.batch_size - tf.reduce_sum(fake_scores / 2 + 0.5)) / config.batch_size

        gradients_of_generator = gen_tape.gradient(Gen_loss, generator_model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(Disc_loss, discriminator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))
    return Gen_loss, Disc_loss, acc_real, acc_fake, gradients_of_generator, gradients_of_discriminator


def train(dataset, epochs):
    lod = np.float32(0.0)
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


            gen_loss, disc_loss, acc_real, acc_fake, grad_gen, grad_dis = train_both(resized_batch, lod)

            with summary_writer.as_default():
                tf.summary.scalar('gen_loss', gen_loss, step=generator_optimizer.iterations)
                tf.summary.scalar('disc_loss', disc_loss, step=discriminator_optimizer.iterations)
                tf.summary.scalar('acc_real', acc_real, step=generator_optimizer.iterations)
                tf.summary.scalar('acc_fake', acc_fake, step=discriminator_optimizer.iterations)
                # tf.summary.scalar('grad_gen', grad_gen, step=generator_optimizer.iterations)
                # tf.summary.scalar('grad_dis', grad_dis, step=discriminator_optimizer.iterations)

            iteration += 1

        # if epoch % 10:
        generate_and_save_images(epoch + 1, config.seed, lod)
        print('lod:', lod)
        print('gen_loss:', gen_loss)
        print('dis_loss:', disc_loss)
        if increase_lod and lod < config.max_lod:
            lod = np.float32(lod + config.lod_increase)

        if (epoch + 1) % config.epochs_per_lod == 0:
            increase_lod = not increase_lod

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def generate_and_save_images(epoch, test_input, lod):
    predictions = generator_model([test_input, lod])

    plt.figure(figsize=(4, 4))
    for i in range(config.num_examples_to_generate):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# def draw_labels(image_batch):
#     plt.figure(figsize=(2, 3))
#     for i in range(6):
#         lod_res = int(2 ** (np.floor(i) + 2))
#         resampled_image = tf.image.resize(image_batch[3], [lod_res, lod_res],
#                                           method=tf.image.ResizeMethod.BILINEAR)
#         fit_for_discriminator = tf.image.resize(resampled_image, [config.resolution, config.resolution],
#                                                 method=tf.image.ResizeMethod.BILINEAR)
#         plt.subplot(2, 3, i + 1)
#         plt.imshow(fit_for_discriminator[:, :, 0] * 127.5 + 127.5, cmap='gray')
#         plt.title('{} x {}'.format(lod_res, lod_res), fontdict={'fontsize': 8})
#         plt.axis('off')
#     plt.savefig('images/upsampled_bilinear_method.png', dpi=1000)
#     plt.show()


# draw_labels(next(iter(train_dataset)))

train(train_dataset, config.EPOCHS)
