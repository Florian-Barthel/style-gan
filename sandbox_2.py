import tensorflow as tf
import numpy as np

resolution=128
dtype=tf.dtypes.float32
num_channels=3


def number_filters(stage):
    fn_result = int(fmap_base / (2.0 ** stage))
    return fn_result


resolution_log2 = int(np.log2(resolution))
num_layers = resolution_log2 - 2
fmap_base = 2 ** (num_layers + 2 + 4)
print(fmap_base)
counter = 0

print('block {}, resolution: {} x {}, filter: {}'.format(counter, 4, 4, 512))

for res in range(3, resolution_log2 + 1):
    counter += 1
    resolution = 2 ** res
    filters = number_filters(res)
    print('block {}, resolution: {} x {}, filter: {}'.format(counter, resolution, resolution, filters))
