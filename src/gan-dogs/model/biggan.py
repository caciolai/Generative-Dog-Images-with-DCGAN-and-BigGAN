import tensorflow as tf
from tensorflow.keras import layers

from .layers import *


class BigGANGenerator(tf.keras.Model):
    def __init__(self, channels, latent_dim, num_classes, sn=True, attn=True, orth=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_upsamples = 4
        self.channels = channels
        self.num_classes = num_classes
        self.attn = attn

        regularizer = OrthogonalRegularizer() if orth else None

        # Sequential structure
        ch = self.channels

        # (128,) -> (4*4*8ch,)
        self.fc = FullyConnected(units=4 * 4 * 8*ch, sn=sn)
        self.embed_reshape = layers.Reshape([ch])
        self.embed = layers.Embedding(num_classes, ch)
        # -> (4, 4, 8ch)
        self.reshape = layers.Reshape((4, 4, 8*ch))

        # -> (8, 8, 8ch)
        self.res1 = ResBlockUp(8*ch, 8*ch, use_bias=False,
                               sn=sn, regularizer=regularizer)

        # -> (16, 16, 4ch)
        self.res2 = ResBlockUp(8*ch, 4*ch, use_bias=False,
                               sn=sn, regularizer=regularizer)

        # -> (32, 32, 2ch)
        self.res3 = ResBlockUp(4*ch, 2*ch, use_bias=False,
                               sn=sn, regularizer=regularizer)
        if attn:
            self.self_attention = SelfAttention(2*ch, sn=sn)

        # -> (64, 64, ch)
        self.res4 = ResBlockUp(2*ch, ch, use_bias=False,
                               sn=sn, regularizer=regularizer)
        self.bn = layers.BatchNormalization(axis=-1)

        # -> (64, 64, 3)
        self.conv = Conv(3, kernel=3, use_bias=False,
                         sn=sn, regularizer=regularizer)

    def call(self, z, y, training=True):
        y = self.embed_reshape(self.embed(y))

        h = self.fc(z, training=training)
        h = self.reshape(h)

        h = self.res1(h, y, training=training)
        h = self.res2(h, y, training=training)
        h = self.res3(h, y, training=training)
        if self.attn:
            h = self.self_attention(h, training=training)

        h = self.res4(h, y, training=training)

        h = self.bn(h, training=training)
        h = tf.nn.leaky_relu(h)
        o = self.conv(h, training=training)
        o = tf.keras.activations.tanh(o)

        return o

    def model(self):
        x = layers.Input(shape=(self.latent_dim))
        y = layers.Input(shape=(1), dtype=tf.int32)
        return tf.keras.Model(inputs=[x, y], outputs=self.call(x, y))

    def summary(self):
        return self.model().summary()


class BigGANDiscriminator(tf.keras.Model):
    def __init__(self, channels, w, h, num_classes, sn=True, attn=True):
        super().__init__()
        self.w = w
        self.h = h
        self.num_classes = num_classes

        self.channels = channels
        ch = self.channels

        # Sequential structure
        # (64, 64, 3) -> (32, 32, ch)
        self.res1 = ResBlockDown(3, ch, use_bias=False, sn=sn)
        if attn:
            self.self_attention = SelfAttention(ch, sn=sn)

        # -> (16, 16, 4ch)
        self.res2 = ResBlockDown(ch, 2*ch, use_bias=False, sn=sn)

        # -> (8, 8, 8ch)
        self.res3 = ResBlockDown(2*ch, 4*ch, use_bias=False, sn=sn)

        # -> (4, 4, 8ch)
        self.res4 = ResBlockDown(4*ch, 8*ch, use_bias=False, sn=sn)

        self.linear = FullyConnected(units=1, sn=sn)
        self.embed = layers.Embedding(
            num_classes, ch*8,
        )

        self.attn = attn

    def call(self, x, y, training=True):
        """
        Implementation of BigGAN discriminator.
        Taken from arXiv:1809.11096v2.
        """
        # squeezing if necessary
        y_shape = tf.shape(y)
        y = tf.reshape(y, shape=[-1])
        emb = self.embed(y)

        h = self.res1(x, training=training)
        if self.attn:
            h = self.self_attention(h, training=training)
        h = self.res2(h, training=training)
        h = self.res3(h, training=training)
        h = self.res4(h, training=training)

        # global sum pooling
        h = tf.nn.leaky_relu(h)
        h = tf.reduce_sum(h, axis=[1, 2])

        linear = self.linear(h, training=training)
        inner = tf.reduce_sum(h*emb, axis=-1, keepdims=True)

        o = linear + inner
        return o

    def model(self):
        x = layers.Input(shape=(self.h, self.w, 3))
        y = layers.Input(shape=(), dtype=tf.int32)
        return tf.keras.Model(inputs=[x, y], outputs=self.call(x, y))

    def summary(self):
        return self.model().summary()
