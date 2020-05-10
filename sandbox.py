import numpy as np
import generator_new
import matplotlib.pyplot as plt
import discriminator_new

num_examples_to_generate = 16
seed = np.random.rand(num_examples_to_generate, 32, 1)
lod_in = np.float32(0.9)

generator_model = generator_new.Generator(resolution=32, num_channels=1)
discriminator_model = discriminator_new.Discriminator(resolution=32, num_channels=1)

plt.figure(figsize=(4, 4))
generated_images = generator_model([seed, lod_in])
predictions = discriminator_model([generated_images, lod_in])
print(predictions)

for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

plt.savefig('images/generated')
plt.show()

print('')
