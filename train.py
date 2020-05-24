import shutil
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

import generator
import discriminator
import config
import loss
import image_utils
import dataset
import save_images

# Setup Environment
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Create Directories
summary_writer = tf.summary.create_file_writer(logdir=config.log_dir)
if not os.path.exists(config.result_folder):
    os.makedirs(config.result_folder)
shutil.copyfile('./config.py', config.result_folder + '/config.py')

# Initialize Models
generator_model = generator.Generator(num_mapping_layers=config.num_mapping_layers,
                                      mapping_fmaps=config.mapping_fmaps,
                                      resolution=config.resolution,
                                      fmap_base=config.fmap_base,
                                      num_channels=config.num_channels,
                                      use_wscale=config.use_wscale)
discriminator_model = discriminator.Discriminator(resolution=config.resolution,
                                                  fmap_base=config.fmap_base,
                                                  use_wscale=config.use_wscale)

# Initialize Optimizer
generator_optimizer = tf.keras.optimizers.Adam(beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = tf.keras.optimizers.Adam(beta_1=0.0, beta_2=0.99, epsilon=1e-8)

gen_var_list = []
disc_var_list = []
var_list_index = 0


@tf.function
def train_generator(latents, lod):
    global var_list_index
    with tf.GradientTape() as gen_tape:
        gen_loss = loss.g_logistic_nonsaturating(generator_model,
                                                 discriminator_model,
                                                 latents=latents,
                                                 lod=lod)

    gen_vars = gen_var_list[var_list_index]
    gradients_of_generator = gen_tape.gradient(gen_loss, gen_vars)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_vars))
    return gen_loss


@tf.function
def train_discriminator(latents, images, lod):
    global var_list_index
    disc_vars = disc_var_list[var_list_index]
    with tf.GradientTape() as disc_tape:
        disc_loss = loss.d_logistic_simplegp(generator_model,
                                             discriminator_model,
                                             lod=lod,
                                             images=images,
                                             latents=latents)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc_vars)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_vars))
    return disc_loss


def init():
    image_dataset = iter(dataset.get_ffhq(256))
    tf.config.experimental_run_functions_eagerly(False)
    lod = 0.0
    while lod <= config.max_lod:
        lod_res = int(2 ** (np.ceil(lod) + 2))
        resized_images = image_utils.resize(next(image_dataset), lod)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            latent = tf.random.normal([config.batch_size, config.latent_size, 1], mean=0.0, stddev=1.0,
                                      dtype=tf.dtypes.float32)
            gen_loss = loss.g_logistic_nonsaturating(generator_model,
                                                     discriminator_model,
                                                     latents=latent,
                                                     lod=np.float32(lod))
            latent = tf.random.normal([config.batch_size, config.latent_size, 1], mean=0.0, stddev=1.0,
                                      dtype=tf.dtypes.float32)
            disc_loss = loss.d_logistic_simplegp(generator_model,
                                                 discriminator_model,
                                                 lod=np.float32(lod),
                                                 images=resized_images,
                                                 latents=latent)

        gen_vars = generator_model.trainable_variables
        disc_vars = discriminator_model.trainable_variables
        gen_var_list.append(gen_vars)
        disc_var_list.append(disc_vars)

        gradients_of_generator = gen_tape.gradient(gen_loss, gen_vars)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, disc_vars)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_vars))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_vars))
        print('Initialize BLock {}x{}'.format(lod_res, lod_res))
        lod += 0.5
    tf.config.experimental_run_functions_eagerly(False)


def train_loop():
    image_dataset = iter(dataset.get_ffhq(4))
    lod = 0.0
    gen_loss = 0
    disc_loss = 0
    increase_lod = False
    global var_list_index
    for epoch in range(1, config.epochs):
        progress_bar = tqdm(range(config.epoch_iterations))
        progress_bar.set_description('Iteration: {}, LoD: {}'.format(epoch * config.epoch_iterations * config.minibatch_repeat, lod))
        for _ in progress_bar:
            image_batch = image_utils.fade_lod(next(image_dataset), lod)
            for _ in range(config.minibatch_repeat):
                latent = tf.random.normal([config.batch_size, config.latent_size, 1], mean=0.0, stddev=1.0,
                                          dtype=tf.dtypes.float32)
                disc_loss = train_discriminator(latents=latent, images=image_batch, lod=np.float32(lod))
                latent = tf.random.normal([config.batch_size, config.latent_size, 1], mean=0.0, stddev=1.0,
                                          dtype=tf.dtypes.float32)
                gen_loss = train_generator(latents=latent, lod=np.float32(lod))

        save_images.generate_and_save_images(generator_model, epoch, lod)
        with summary_writer.as_default():
            tf.summary.scalar('gen_loss', gen_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

        if epoch % config.iterations_per_lod == 0:
            if config.reset_optimizer and not increase_lod:
                print('Reset Optimizer')
                for var in generator_optimizer.variables():
                    var.assign(tf.zeros_like(var))
                for var in discriminator_optimizer.variables():
                    var.assign(tf.zeros_like(var))
            increase_lod = not increase_lod
            var_list_index += 1

        if increase_lod and lod < config.max_lod:
            lod = np.around(lod + config.lod_increase, decimals=config.lod_decimals)
            if epoch % config.iterations_per_lod == 0:
                print('Change Dataset')
                image_dataset = iter(dataset.get_ffhq(int(2 ** (np.ceil(lod) + 2))))


init()
train_loop()
