import tensorflow as tf
import generator
import generator_loss
import discriminator
import discriminator_loss
import os
import config
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime

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

generator_model = generator.generator_model(mapping_fmaps=config.resolution,
                                            resolution=config.resolution,
                                            fmap_base=32,
                                            num_channels=1)
discriminator_model = discriminator.discriminator_model(resolution=config.resolution,
                                                        fmap_base=32,
                                                        number_of_channels=1)

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
        latents = tf.random.normal([config.batch_size, config.resolution, 1])
        lods = np.full((config.batch_size, 1), lod)
        fake_images_out = generator_model([latents, lods], training=True)
        real_scores_out = discriminator_model([images, lods], training=True)
        fake_scores_out = discriminator_model([fake_images_out, lods], training=True)

        gen_loss_tensor = generator_loss.cross_entropy_loss(fake_scores_out=fake_scores_out)
        disc_loss_tensor = discriminator_loss.cross_entropy_loss(real_scores_out=real_scores_out,
                                                                 fake_scores_out=fake_scores_out)

        gen_loss = tf.reduce_mean(gen_loss_tensor)
        disc_loss = tf.reduce_mean(disc_loss_tensor)

        acc_real = tf.reduce_sum(real_scores_out) / config.batch_size
        acc_fake = (config.batch_size - tf.reduce_sum(fake_scores_out)) / config.batch_size

    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))
    return gen_loss, disc_loss, acc_real, acc_fake


@tf.function
def train_only_discriminator(images, lod):
    with tf.GradientTape() as disc_tape:
        lods = np.full((config.batch_size, 1), lod)
        real_scores_out = discriminator_model([images, lods], training=True)
        disc_loss_tensor = discriminator_loss.D_logistic_simplegp_only_real(real_scores_out=real_scores_out)
        disc_loss = tf.reduce_mean(disc_loss_tensor)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))


def train(dataset, epochs):
    lod = 0.0
    gen_loss = 0
    disc_loss = 0
    increase_lod = False
    iteration = 0
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            lod_res = int(2 ** (np.floor(lod) + 2))
            resize1 = tf.image.resize(image_batch, [lod_res, lod_res],
                                      method=tf.image.ResizeMethod.AREA)
            resize2 = tf.image.resize(resize1, [config.resolution, config.resolution],
                                      method=tf.image.ResizeMethod.AREA)

            gen_loss, disc_loss, acc_real, acc_fake = train_both(resize2, lod)

            with summary_writer.as_default():
                tf.summary.scalar('gen_loss', gen_loss, step=generator_optimizer.iterations)
                tf.summary.scalar('disc_loss', disc_loss, step=discriminator_optimizer.iterations)
                tf.summary.scalar('acc_real', acc_real, step=generator_optimizer.iterations)
                tf.summary.scalar('acc_fake', acc_fake, step=discriminator_optimizer.iterations)

            iteration += 1

        # if epoch % 10:
        generate_and_save_images(epoch + 1,
                                 config.seed,
                                 lod)
        print('lod:', lod)
        print('gen_loss:', gen_loss)
        print('dis_loss:', disc_loss)
        if increase_lod:
            lod += 0.02

        if (epoch + 1) % 50 == 0:
            increase_lod = not increase_lod

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def generate_and_save_images(epoch, test_input, lod):
    lods = np.full((test_input.shape[0], 1), lod)
    predictions = generator_model([test_input, lods], training=False)

    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def draw_labels(image_batch):
    plt.figure(figsize=(2, 3))
    for i in range(6):
        lod_res = int(2 ** (np.floor(i) + 2))
        resampled_image = tf.image.resize(image_batch[3], [lod_res, lod_res],
                                          method=tf.image.ResizeMethod.AREA)
        fit_for_discriminator = tf.image.resize(resampled_image, [config.resolution, config.resolution],
                                                method=tf.image.ResizeMethod.AREA)
        plt.subplot(2, 3, i + 1)
        plt.imshow(fit_for_discriminator[:, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.title('{} x {}'.format(lod_res, lod_res), fontdict={'fontsize': 8})
        plt.axis('off')
    plt.savefig('images/label_resolutions_disc_input.png', dpi=1000)
    plt.show()

# draw_labels(next(iter(train_dataset)))

train(train_dataset, config.EPOCHS)
