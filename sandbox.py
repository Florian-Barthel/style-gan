import tensorflow as tf

lod_in = tf.constant(5)
lod = 5
def fromrgb(x):  # res = 2..resolution_log2
    return tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), activation=tf.nn.relu)(x)


def block():
    sub_model = tf.keras.models.Sequential()
    sub_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu))
    return sub_model


def lerp_clip(a, b, t):
    """Linear interpolation with clip."""
    with tf.name_scope("LerpClip"):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


img = tf.constant([[[[5]]]])

x = fromrgb(img)
x = block(x)
y = fromrgb(img)
x = tf.keras.layers.Lambda(lerp_clip)(x, y, lod_in - lod)