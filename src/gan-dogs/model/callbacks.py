import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3

from .utils import scale_images, calculate_fid, plot_imgs_grid


class FIDCallback(tf.keras.callbacks.Callback):
    """Callback to calculate FID score during training
    """

    def __init__(self, dataset, num_classes, period=5):
        super().__init__()

        self.period = period
        self.dataset = iter(dataset)
        self.num_classes = num_classes
        self.inception_model = InceptionV3(
            include_top=False, pooling='avg', input_shape=(299, 299, 3))

    def compute_fid(self):
        num_classes = self.num_classes

        # real images
        real_images, real_labels = next(self.dataset)
        num_images = real_images.shape[0]
        latent_dim = self.model.latent_dim

        # generated images
        latent_samples = tf.random.truncated_normal(
            shape=(num_images, latent_dim))
        random_labels = tf.math.floor(
            num_classes * tf.random.uniform((num_images, 1)))

        inputs = (latent_samples, random_labels)

        generated_images = self.model(inputs, training=False)

        # resize images
        real_images = scale_images(real_images, (299, 299, 3))
        generated_images = scale_images(generated_images, (299, 299, 3))

        # calculate fid
        fid = calculate_fid(self.inception_model,
                            real_images, generated_images)
        return fid

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period != 0:
            return

        fid = self.compute_fid()
        print(f"\n\n === FID: {fid} ===\n")


class PlotImagesCallback(tf.keras.callbacks.Callback):
    """Callback to plot images during training (evaluation pass)
    """

    def __init__(self, num_classes, n_images=16, period=10):
        super().__init__()

        self.num_classes = num_classes
        self.n_images = n_images
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period != 0:
            return

        num_classes = self.num_classes
        latent_dim = self.model.latent_dim
        latent_sample = tf.random.truncated_normal(
            shape=(self.n_images, latent_dim))
        random_labels = tf.math.floor(
            num_classes * tf.random.uniform((self.n_images, 1)))

        inputs = (latent_sample, random_labels)

        imgs = self.model(inputs, training=False)
        plot_imgs_grid(imgs)


def get_callbacks(train_dataset, checkpoint_path=None):
    """Return selected list of callbacks to employ during training.

    Args:
        train_dataset (tf.data.Dataset): training dataset
        checkpoint_path (str, optional): path where to save model training progress. Defaults to None.

    Returns:
        List[tf.keras.callbacks.Callback]: list of callbacks
    """
    callbacks = []

    if checkpoint_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, verbose=0, mode='auto', save_freq='epoch', options=None
            )
        )

    callbacks.append(
        FIDCallback(
            train_dataset, period=10
        )
    )

    callbacks.append(
        PlotImagesCallback(
            n_images=16, period=10
        )
    )

    return callbacks
