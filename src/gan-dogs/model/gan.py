import tensorflow as tf

from .loss import RaLSGANDiscriminatorLoss, RaLSGANGeneratorLoss, SGANDiscriminatorLoss, SGANGeneratorLoss, WGANDiscriminatorLoss, WGANGeneratorLoss
from .dcgan import DCGANDiscriminator, DCGANGenerator
from .biggan import BigGANDiscriminator, BigGANGenerator
from ..data.configuration import get_configuration


class GAN(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        num_classes,
        d_steps,
        gp_lambda
    ):
        """Generative Adversarial Network.

        Args:
            discriminator (tf.keras.Model): discriminator model
            generator (tf.keras.Model): generator model
            latent_dim (int): dimension of latent space
            num_classes (int): number of classes
            d_steps (int): number of discriminator training steps per generator training step
            gp_lambda (float): gradient penalty coefficient [description]
        """
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.d_steps = d_steps
        self.gp_lambda = gp_lambda

    def compile(self, d_optim, g_optim,
                d_loss_fn, g_loss_fn):
        """Compiles the GAN, using the standard GAN loss implemented as a BinaryCrossentropy from logits (non-sigmoid discriminator output).

        Args:
            d_optim (tf.keras.Optimizer): Optimizer for the discriminator
            g_optim (tf.keras.Optimizer): Optimizer for the generator
            d_loss_fn (tf.keras.Loss): Loss function for the discriminator
            g_loss_fn (tf.keras.Loss): Loss Function for the generator
        """
        super().compile()
        self.d_optim = d_optim
        self.g_optim = g_optim
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def call(self, batch, training=True):
        """Generates some images from a batch of latent codes and classes.

        Args:
            batch ((tf.Tensor, tf.Tensor)): batch of latent codes and classes
            training (bool, optional):  Defaults to True.

        Returns:
            tf.Tensor: generated images
        """
        x, y = batch
        return self.generator(x, y, training=training)

    def gradient_penalty(self, real_images, fake_images):
        """
        Calculates the gradient penalty.
        This loss is calculated on an interpolated image and added to the discriminator loss.

        Args:
            real_images (tf.Tensor): real images
            fake_images (tf.Tensor): generated images
        Returns:
            float: gradient penalty.
        """
        bs = tf.shape(real_images)[0]
        epsilon = tf.random.uniform(
            shape=[bs, 1, 1, 1], minval=0.0, maxval=1.0)
        interpolated = epsilon * real_images + (1 - epsilon) * fake_images

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    def compute_d_loss(self, real_output, fake_output, real_images=None, fake_images=None):
        """Computes discriminator loss.

        Args:
            real_output (tf.Tensor): discriminator output on real images
            fake_output (tf.Tensor): discriminator output on fake images
            real_images (tf.Tensor, optional): real images (from dataset). Defaults to None.
            fake_images (tf.Tensor, optional): fake images (from generator). Defaults to None.

        Returns:
            float: discriminator loss
        """
        d_loss = self.d_loss_fn(real_output, fake_output)
        if isinstance(self.d_loss_fn, WGANDiscriminatorLoss):
            assert real_images is not None and fake_images is not None

            gp = self.gradient_penalty(real_images, fake_images)
            d_loss = d_loss + self.gp_lambda * gp

        return d_loss

    def compute_g_loss(self, real_output, fake_output):
        """Computes the generator loss

        Args:
            real_output (tf.Tensor): discriminator output on real images
            fake_output (tf.Tensor): discriminator output on fake images

        Returns:
            float: Generator loss
        """
        if isinstance(self.g_loss_fn, RaLSGANGeneratorLoss):
            g_loss = self.g_loss_fn(real_output, fake_output)
        else:
            g_loss = self.g_loss_fn(None, fake_output)

        return g_loss

    def train_step(self, batch):
        """Performs one step of GAN training:
            1. Sample a batch of real images
            2. Train the discriminator for d_steps steps
                2.1. Sample a batch of priors from latent space
                2.2. Generate a batch of fake images from those priors
                2.3. Obtain the discriminator prediction on both the real and fake images
                2.4. Optimize the discriminator based on its predictions
            3. Train the generator
                3.1. Sample a batch of priors from latent space
                3.2. Generate a batch of fake images from those priors
                3.3. Obtain the discriminator prediction on the fake images
                3.4. Optimize the generator based on the discriminator predictions

        Args:
            batch (Tuple[tf.Tensor, tf.Tensor]): training batch of (real images, real labels)

        Returns:
            dict: losses to be recorded in training history
        """
        real_images, real_labels = batch
        bs = tf.shape(real_images)[0]

        d_loss_total = 0.0
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(bs, self.latent_dim)
            )
            random_labels = tf.math.floor(
                self.num_classes * tf.random.uniform(shape=[bs, 1]))

            with tf.GradientTape() as tape:
                fake_images = self.generator(
                    random_latent_vectors, random_labels, training=True)
                real_output = self.discriminator(
                    real_images, real_labels, training=True)
                fake_output = self.discriminator(
                    fake_images, random_labels, training=True)

                d_loss = self.compute_d_loss(real_output, fake_output)
                d_loss = d_loss / self.d_steps
                d_loss_total += d_loss

            d_grad = tape.gradient(
                d_loss, self.discriminator.trainable_variables)
            self.d_optim.apply_gradients(
                zip(d_grad, self.discriminator.trainable_variables)
            )

        random_latent_vectors = tf.random.normal(
            shape=(bs, self.latent_dim)
        )
        random_labels = tf.math.floor(
            self.num_classes * tf.random.uniform(shape=[bs, 1]))

        with tf.GradientTape() as tape:
            fake_images = self.generator(
                random_latent_vectors, random_labels, training=True)
            fake_output = self.discriminator(
                fake_images, random_labels, training=True)
            g_loss = self.compute_g_loss(real_output, fake_output)

        g_grad = tape.gradient(g_loss, self.generator.trainable_variables)

        self.g_optim.apply_gradients(
            zip(g_grad, self.generator.trainable_variables)
        )

        return {"d_loss": d_loss_total, "g_loss": g_loss}


def build_gan(use_model, num_classes, img_width, img_height):
    """Builds a GAN model.

    Args:
        use_model (str): Either 'DCGAN' or 'BigGAN'
        num_classes (int): Number of classes (dog breeds)
        img_width (int): Width of training images
        img_height (int): Height of training images

    Raises:
        NotImplementedError: in case of different use_model

    Returns:
        GAN: built GAN model
    """
    CONFIG = get_configuration()

    if use_model == 'DCGAN':
        print("Building discriminator...")
        discriminator = DCGANDiscriminator()
        print("Done.")
        print("Building generator...")
        generator = DCGANGenerator()
        print("Done.")

        print("Building GAN...")
        gan = GAN(
            discriminator=discriminator,
            generator=generator,
            latent_dim=CONFIG["latent_dim"],
            num_classes=num_classes,
            d_steps=CONFIG["d_steps"],
            gp_lambda=CONFIG["gp_lambda"]
        )
        print("Done.")
    elif use_model == 'BigGAN':
        print("Building discriminator...")
        discriminator = BigGANDiscriminator(
            channels=CONFIG["channels"], w=img_width, h=img_height, num_classes=num_classes,
            sn=CONFIG["use_spec_norm"], attn=CONFIG["use_attention"]
        )
        print("Done.")
        print("Building generator...")
        generator = BigGANGenerator(
            channels=CONFIG["channels"], latent_dim=CONFIG["latent_dim"], num_classes=num_classes,
            sn=CONFIG["use_spec_norm"], attn=CONFIG["use_attention"], orth=CONFIG["use_orth_reg"]
        )
        print("Done.")

        print("Building GAN...")
        gan = GAN(
            discriminator=discriminator,
            generator=generator,
            latent_dim=CONFIG["latent_dim"],
            num_classes=num_classes,
            d_steps=CONFIG["d_steps"],
            gp_lambda=CONFIG["gp_lambda"]
        )
        print("Done.")
    else:
        raise NotImplementedError

    return gan


def compile(gan):
    """Compiles a built GAN model

    Args:
        gan (GAN): built GAN model

    Returns:
        GAN: compiled GAN model
    """
    CONFIG = get_configuration()

    lr_d = CONFIG["lr_d"]
    lr_g = CONFIG["lr_g"]
    beta_1 = CONFIG["beta1"]
    beta_2 = CONFIG["beta2"]

    if CONFIG["exp_decay"]:
        decay_rate = CONFIG["decay_rate"]
        decay_steps = CONFIG["decay_steps"]

        lr_d = tf.keras.optimizers.schedules.ExponentialDecay(
            lr_d,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )

        lr_g = tf.keras.optimizers.schedules.ExponentialDecay(
            lr_g,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )

    d_optim = tf.keras.optimizers.Adam(
        learning_rate=lr_d, beta_1=beta_1, beta_2=beta_2
    )

    g_optim = tf.keras.optimizers.Adam(
        learning_rate=lr_g, beta_1=beta_1, beta_2=beta_2
    )

    if CONFIG["loss"] == "sgan":
        d_loss_fn = SGANDiscriminatorLoss()
        g_loss_fn = SGANGeneratorLoss()
    elif CONFIG["loss"] == "wgan":
        d_loss_fn = WGANDiscriminatorLoss()
        g_loss_fn = WGANGeneratorLoss()
    elif CONFIG["loss"] == "ralsgan":
        d_loss_fn = RaLSGANDiscriminatorLoss()
        g_loss_fn = RaLSGANGeneratorLoss()

    print("Compiling GAN...")
    gan.compile(
        d_optim=d_optim,
        g_optim=g_optim,
        d_loss_fn=d_loss_fn,
        g_loss_fn=g_loss_fn
    )
    print("Done.")

    return gan
