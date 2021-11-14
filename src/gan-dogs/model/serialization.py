import tensorflow as tf
from tensorflow.python.framework.tensor_conversion_registry import get

from .gan import build_gan


def load_model(use_model, num_classes, latent_dim, img_width, img_height, checkpoint_path):
    """Load model from checkpoint on disk.

    Args:
        use_model (str): name of model (DCGAN, BigGAN)
        num_classes (int): number of classes
        latent_dim (int): dimension of latent space
        img_width (int): width of incoming images
        img_height (int): height of incoming images
        checkpoint_path (str): path of the checkpoint to load on disk

    Returns:
        GAN: loaded GAN model
    """

    gan = build_gan(use_model, num_classes, img_width, img_height)

    # testing
    random_latent_vectors = tf.random.normal(
        shape=(1, latent_dim)
    )
    random_labels = tf.math.floor(num_classes * tf.random.uniform((1, 1)))

    inputs = (random_latent_vectors, random_labels)
    _ = gan(inputs)

    path = checkpoint_path
    # path = "/content/drive/My Drive/VisionPerception/SavedModel/BigGAN_dogs_64_450_epochs/checkpoint.ckpt"
    gan.load_weights(path)

    return gan


def save_model(gan, save_path):
    """Save GAN model to disk

    Args:
        gan (GAN): GAN model
        save_path (str): path where to save the model weights
    """
    gan.save_weights(save_path)
