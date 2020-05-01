import os
import tensorflow as tf
import numpy as np


def blur2d(x, f=[1, 2, 1], normalize=True, flip=False, stride=1):
    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0, 0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, 1, stride, stride]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format='NCHW')
    x = tf.cast(x, orig_dtype)
    return x


def upscale2d(x, factor=2, gain=1):
    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = tf.shape(x)

    reshaped1 = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    tiled = tf.tile(reshaped1, [1, factor, factor, 1, 1, 1])
    reshaped2 = tf.reshape(tiled, [-1, s[1] * factor, s[2] * factor, s[3]])
    return reshaped2


def downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, 1, factor, factor]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')