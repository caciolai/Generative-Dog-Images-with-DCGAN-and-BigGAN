import tensorflow as tf

from .utils import noisy_labels, smooth_fake_labels, smooth_real_labels, CONFIG


class SGANDiscriminatorLoss(tf.keras.losses.Loss):

    def __init__(self):
        """Standard GAN loss for discriminator.
        """
        super().__init__()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, real_output, fake_output):
        """Loss for the discriminator. 
        Applies technique from GAN hacks to stabilize training:
            - Label smoothing
            - Label noise

        Args:
            real_output (tf.Tensor): output of discriminator on real images
            fake_output (tf.Tensor): output of discriminator on fake images

        Returns:
            float: discriminator loss
        """

        # Real images must be predicted 1 (noised and smoothed)
        real_labels = tf.ones_like(real_output)
        if CONFIG["smooth_labels"]:
            real_labels = noisy_labels(real_labels, CONFIG["label_noise"])
            real_labels = smooth_real_labels(real_labels)

        # Fake images must be predicted 0 (noised and smoothed)
        fake_labels = tf.zeros_like(fake_output)
        if CONFIG["smooth_labels"]:
            fake_labels = noisy_labels(fake_labels, CONFIG["label_noise"])
            fake_labels = smooth_fake_labels(fake_labels)

        real_loss = self.bce(real_labels, real_output)
        fake_loss = self.bce(fake_labels, fake_output)

        total_loss = real_loss + fake_loss
        return total_loss


class SGANGeneratorLoss(tf.keras.losses.Loss):
    def __init__(self):
        """Standard GAN loss for generator.
        """
        super().__init__()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, real_output, fake_output):
        """Loss for the generator. The generator must fool the discriminator,
        making it predict fake images as real.

        Args:
            real_output (tf.Tensor): output of the discriminator on real images 
                        (actually not used, just to comply with interface function signature)
            fake_output (tf.Tensor): output of the discriminator on fake (generated) images

        Returns:
            float: generator loss
        """
        loss = self.bce(tf.ones_like(fake_output), fake_output)
        return loss


class WGANDiscriminatorLoss(tf.keras.losses.Loss):
    def __init__(self) -> None:
        """Wasserstein loss for the 'critic' from `Wasserstein GAN` (https://arxiv.org/abs/1701.07875).
        """
        super().__init__()

    def call(self, real_output, gen_output):
        # loss for the output of the discriminator on real images
        real_loss = tf.reduce_mean(real_output)
        # loss for the output of the discriminator on generated images
        gen_loss = tf.reduce_mean(gen_output)
        loss = gen_loss - real_loss
        return loss


class WGANGeneratorLoss(tf.keras.losses.Loss):
    def __init__(self) -> None:
        """Wasserstein loss for the generator from `Wasserstein GAN` (https://arxiv.org/abs/1701.07875).
        """
        super().__init__()

    def call(self, real_output, gen_output):

        loss = -tf.reduce_mean(gen_output)
        return loss


class RaLSGANGeneratorLoss(tf.keras.losses.Loss):
    def __init__(self) -> None:
        """Loss for Relativistic average Least Square GAN (arXiv:1901.02474).
        """
        super().__init__()

    def call(self, real_output, fake_output):

        real_loss = tf.reduce_mean(
            real_output - tf.reduce_mean(fake_output) + 1)**2
        fake_loss = tf.reduce_mean(
            fake_output - tf.reduce_mean(real_output) - 1)**2

        loss = (real_loss + fake_loss) / 2

        return loss


class RaLSGANDiscriminatorLoss(tf.keras.losses.Loss):
    def __init__(self) -> None:
        """Loss for Relativistic average Least Square GAN (arXiv:1901.02474).
        """
        super().__init__()

    def call(self, real_output, fake_output):

        real_loss = tf.reduce_mean(
            real_output - tf.reduce_mean(fake_output) - 1)**2
        fake_loss = tf.reduce_mean(
            fake_output - tf.reduce_mean(real_output) + 1)**2

        loss = (real_loss + fake_loss) / 2

        return loss
