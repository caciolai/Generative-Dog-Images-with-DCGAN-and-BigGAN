import tensorflow as tf
from tensorflow.keras import layers

from ..data.configuration import get_configuration


class DCGANGenerator(tf.keras.Model):
    def __init__(self):
        """Generator for the implemented DCGAN model.
        """
        super().__init__()

        CONFIG = get_configuration()

        self.z_dim = CONFIG["latent_dim"]

        self.dense = layers.Dense(8*8*512, use_bias=False)
        self.bn = layers.BatchNormalization(axis=-1)
        self.reshape = layers.Reshape([8, 8, 512])

        # 8 -> 16
        self.deconv1 = layers.Conv2DTranspose(
            256, 5, 1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization(axis=-1)

        # 16 -> 32
        self.deconv2 = layers.Conv2DTranspose(
            128, 5, 2, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization(axis=-1)

        # 32 -> 64
        self.deconv3 = layers.Conv2DTranspose(
            64, 5, 2, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization(axis=-1)

        self.final_deconv = layers.Conv2DTranspose(
            3, 5, 2, padding='same', use_bias=False, activation='tanh')

    def call(self, x, y, training=True):
        # 128 -> 8*8*512 -> 8, 8, 512
        h = self.dense(x)
        h = self.bn(h)
        h = tf.nn.relu(h)
        h = self.reshape(h)

        # 8 -> 16
        h = self.deconv1(h)
        h = self.bn1(h)
        h = tf.nn.relu(h)

        # 16 -> 32
        h = self.deconv2(h)
        h = self.bn2(h)
        h = tf.nn.relu(h)

        # 32 -> 64
        h = self.deconv3(h)
        h = self.bn3(h)
        h = tf.nn.relu(h)

        o = self.final_deconv(h)
        return o

    def model(self):
        x = layers.Input(shape=(self.z_dim))
        y = layers.Input(shape=(1), dtype=tf.int32)
        return tf.keras.Model(inputs=[x, y], outputs=self.call(x, y))

    def summary(self):
        return self.model().summary()


class DCGANDiscriminator(tf.keras.Model):
    def __init__(self, img_width, img_height):
        """Discriminator for the implemented DCGAN model.

        Args:
            img_width (int): width of the incoming images
            img_height (int): height of the incoming images
        """
        super().__init__()
        self.w, self.h = img_width, img_height

        self.conv1 = layers.Conv2D(
            64, 5, 2, padding='same', input_shape=[self.w, self.h, 3])
        self.dp1 = layers.Dropout(0.3)

        # 32 -> 16
        self.conv2 = layers.Conv2D(128, 5, 2, padding='same')
        self.dp2 = layers.Dropout(0.3)

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1)

    def call(self, x, y, training=True):

        # 64 -> 32
        h = self.conv1(x)
        h = tf.nn.leaky_relu(h)
        h = self.dp1(h)

        # 32 -> 16
        h = self.conv2(h)
        h = tf.nn.leaky_relu(h)
        h = self.dp2(h)

        h = self.flatten(h)
        o = self.dense(h)

        return o

    def model(self):
        x = layers.Input(shape=[self.h, self.w, 3])
        y = layers.Input(shape=(1), dtype=tf.int32)
        return tf.keras.Model(inputs=[x, y], outputs=self.call(x, y))

    def summary(self):
        return self.model().summary()
