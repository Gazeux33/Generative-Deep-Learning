import tensorflow as tf
from tensorflow.keras import callbacks
import os
import matplotlib.pyplot as plt


class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img, latent_dim, save_dir):
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim)
        )
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = generated_images.numpy()

        for i, image in enumerate(generated_images):
            plt.imshow(image, cmap="gray")
            plt.axis('off')
            plt.savefig(os.path.join(self.save_dir, f'img_{epoch}_{i}.png'))
            plt.close()