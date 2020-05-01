import tensorflow as tf
from PIL import ImageDraw, Image
import generator
import matplotlib.pyplot as plt
import numpy as np
import discriminator

num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, 32, 1])

generator_model = generator.generator_model()
discriminator_model = discriminator.discriminator_model()

plt.figure(figsize=(4, 4))
generated_images = generator_model(seed, training=False)
predictions = discriminator_model(generated_images, training=False)
print(predictions)
for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

plt.savefig('images/generated')
plt.show()

print('')
