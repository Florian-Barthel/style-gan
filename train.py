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
import persistence
import metrics
from lod import LoD

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
models_folder = config.result_folder + '/models/'
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

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
    gen_vars = gen_var_list[var_list_index]
    with tf.GradientTape() as gen_tape:
        global generator_model
        global discriminator_model
        gen_loss = loss.g_logistic_nonsaturating(generator_model,
                                                 discriminator_model,
                                                 latents=latents,
                                                 lod=lod)

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
    tf.config.experimental_run_functions_eagerly(True)
    lod = 0.0
    while lod <= config.max_lod:
        lod_res = int(2 ** (np.ceil(lod) + 2))
        batch_size = config.minibatch_dict[lod_res]
        image_dataset = iter(dataset.get_ffhq(lod_res, batch_size))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            latent = tf.random.normal([batch_size, config.latent_size, 1], dtype=tf.float32)
            gen_loss = loss.g_logistic_nonsaturating(generator_model,
                                                     discriminator_model,
                                                     latents=latent,
                                                     lod=np.float32(lod))
            latent = tf.random.normal([batch_size, config.latent_size, 1], dtype=tf.float32)
            disc_loss = loss.d_logistic_simplegp(generator_model,
                                                 discriminator_model,
                                                 lod=np.float32(lod),
                                                 images=next(image_dataset),
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
    lod = LoD(config.resolution)
    gen_loss = 0
    disc_loss = 0
    num_images = 0
    increase_lod = False
    current_iterations = 0
    batch_size = config.minibatch_dict[4]
    image_dataset = iter(dataset.get_ffhq(4, batch_size))
    image_dataset_eval = iter(dataset.get_ffhq(config.resolution, batch_size))
    global var_list_index
    for iteration in range(1, config.epochs):
        progress_bar = tqdm(range(config.epoch_iterations))
        progress_bar.set_description(
            'Iteration: {}, LoD: {}'.format(iteration * config.epoch_iterations * config.minibatch_repeat,
                                            lod.get_value()))

        for _ in progress_bar:
            image_batch = image_utils.fade_lod(next(image_dataset), lod.get_value())
            for _ in range(config.minibatch_repeat):
                latent = tf.random.normal([batch_size, config.latent_size, 1], dtype=tf.float32)
                disc_loss = train_discriminator(latents=latent, images=image_batch, lod=lod.get_value())
                latent = tf.random.normal([batch_size, config.latent_size, 1], dtype=tf.float32)
                gen_loss = train_generator(latents=latent, lod=lod.get_value())
                num_images += batch_size

        current_iterations += 1
        save_images.generate_and_save_images(generator_model, num_images, lod.get_value(), iteration)
        with summary_writer.as_default():
            tf.summary.scalar('gen_loss', gen_loss, step=iteration)
            tf.summary.scalar('disc_loss', disc_loss, step=iteration)
            tf.summary.scalar('lod', lod.get_value(), step=num_images)

        if iteration % config.evaluation_interval == 0:
            fid_score = metrics.FID(generator_model, image_dataset_eval, lod.get_value(), batch_size)
            tf.summary.scalar('FID', fid_score, step=iteration)
            print('FID:', fid_score)
            # persistence.save_pkl(generator_model, 'gen', iteration)
            # generator_model.save_weights(models_folder + '/gen_model_at_iteration{:04d}'.format(iteration))
            # with open("models_folder + '/gen_model_at_iteration{:04d}'.format(iteration).json", "w") as json_file:
            #     json_file.write(model_json)
            # generator_model.save_model(models_folder + '/gen_model_at_iteration{:04d}'.format(iteration))
            # tf.keras.models.save_model(discriminator_model, models_folder + '/disc_model_at_iteration{:04d}'.format(iteration))

        if current_iterations >= config.iterations_per_lod_dict[lod.get_resolution()]:
            if not increase_lod:
                lod.increase_resolution()
            increase_lod = not increase_lod
            var_list_index += 1
            current_iterations = 0
            lod.round()

        if increase_lod:
            lod.increase_value(steps=config.iterations_per_lod_dict[lod.get_resolution()])
            if current_iterations == 0:
                res = lod.get_resolution()
                batch_size = config.minibatch_dict[res]
                print('Change Dataset')
                image_dataset = iter(dataset.get_ffhq(res=res, batch_size=batch_size))
                if config.reset_optimizer:
                    print('Reset Optimizer')
                    for var in generator_optimizer.variables():
                        var.assign(tf.zeros_like(var))
                    for var in discriminator_optimizer.variables():
                        var.assign(tf.zeros_like(var))


init()
train_loop()
