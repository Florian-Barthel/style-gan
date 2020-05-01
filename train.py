import tensorflow as tf
import generator
import generator_loss
import discriminator
import discriminator_loss
import os
import config
import time
import matplotlib.pyplot as plt


print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('uint8')
train_images = (train_images - 127.5) / 127.5
train_images = tf.image.resize(train_images, (config.resolution, config.resolution))

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(config.BUFFER_SIZE).batch(config.batch_size)

generator_model = generator.generator_model(mapping_fmaps=config.resolution, resolution=config.resolution)
discriminator_model = discriminator.discriminator_model(resolution=config.resolution)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator_model,
                                 discriminator=discriminator_model)


@tf.function
def train_step(images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_loss = generator_loss.G_wgan(discriminator_model, generator_model, config.batch_size, config.resolution)
        disc_loss = discriminator_loss.D_wgan(discriminator_model, generator_model, config.batch_size, images, config.resolution)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))
    return gen_loss, disc_loss


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            # print(gen_loss, disc_loss)
            real_images = image_batch
            generate_and_save_images(epoch + 1,
                                     config.seed,
                                     real_images)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def generate_and_save_images(epoch, test_input, real_image):
    predictions = generator_model(test_input, training=False)
    # print(tf.reduce_sum(discriminator_model(predictions, training=False)))
    # print(tf.reduce_sum(discriminator_model(real_image, training=False)))

    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


train(train_dataset, config.EPOCHS)
