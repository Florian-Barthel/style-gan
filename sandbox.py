import numpy as np

f = [1, 2, 1]

f = np.array(f, dtype=np.float32)
if f.ndim == 1:
    f = f[:, np.newaxis] * f[np.newaxis, :]
assert f.ndim == 2

f /= np.sum(f)
f = f[:, :, np.newaxis, np.newaxis]
f = np.tile(f, [1, 1, int(x.shape[3]), 1])


print(f.shape)
print(np.sum(f))