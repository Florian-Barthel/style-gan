import tensorflow as tf
import numpy as np


@tf.function
def resize(image_batch, lod):
    lod_res = int(2 ** (np.ceil(lod) + 2))
    resized_batch = tf.image.resize(image_batch, [lod_res, lod_res],
                                    method=tf.image.ResizeMethod.AREA)

    # Fade labels with lod
    lod_remainder = lod - int(lod)
    if lod_remainder > 0:
        resized_batch_low = tf.image.resize(image_batch, [int(lod_res / 2), int(lod_res / 2)],
                                            method=tf.image.ResizeMethod.AREA)
        resized_batch_low = tf.image.resize(resized_batch_low, [lod_res, lod_res],
                                            method=tf.image.ResizeMethod.AREA)
        resized_batch = resized_batch + (resized_batch_low - resized_batch) * (1 - lod_remainder)
    return resized_batch


@tf.function
def fade_lod(image_batch, lod):
    lod_remainder = lod - int(lod)
    if lod_remainder == 0:
        return image_batch
    else:
        lod_res = int(2 ** (np.ceil(lod) + 2))
        downsampled = tf.image.resize(image_batch, [int(lod_res / 2), int(lod_res / 2)],
                                            method=tf.image.ResizeMethod.AREA)
        downsampled = tf.image.resize(downsampled, [lod_res, lod_res],
                                            method=tf.image.ResizeMethod.AREA)
        faded_batch = image_batch + (downsampled - image_batch) * (1 - lod_remainder)
    return faded_batch
