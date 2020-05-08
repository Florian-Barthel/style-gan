import discriminator_new
import numpy as np

model = discriminator_new.Discriminator(num_channels=1, resolution=32)

batchsize = 10
lod = 1.0

res = int(2 ** (lod + 2))
images = np.zeros([batchsize, res, res, 1])
scores = model([images, lod])
print(scores)