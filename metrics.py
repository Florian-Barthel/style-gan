import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
import config
from tqdm import tqdm

model = InceptionV3(include_top=False, pooling='avg', input_shape=(config.resolution, config.resolution, 3))

def FID(generator, dataset, lod, batch_size):
    num_batches = config.fid_num_images // batch_size
    activations_gen = np.empty([batch_size * num_batches, 2048], dtype=np.float32)
    activations_real = np.empty([batch_size * num_batches, 2048], dtype=np.float32)
    progress_bar = tqdm(range(num_batches))
    progress_bar.set_description('calculate FID score')
    for i in progress_bar:
        begin = i * batch_size
        end = min(begin + batch_size, num_batches * batch_size)
        latent = tf.random.normal([batch_size, config.latent_size, 1], dtype=tf.float32)
        fake_images_out = generator([latent, lod], trainable=False)
        resized_fake = tf.image.resize(fake_images_out, [config.resolution, config.resolution],
                                       method=tf.image.ResizeMethod.AREA)
        activations_gen[begin:end] = _get_inception_v3_output(resized_fake)

        activations_real[begin:end] = _get_inception_v3_output(next(dataset))
    return _calculate_fid(activations_gen, activations_real)


def _calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


@tf.function
def _get_inception_v3_output(images):
    return model.predict(images)