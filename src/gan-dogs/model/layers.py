import tensorflow as tf
from tensorflow.keras import layers


class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, beta=1e-04):
        self.beta = beta

    def __call__(self, w):
        """
        Orthogonal regularization modified to relax the constraint while
        still imparting the desired smoothness to our models. 
        This modified version removes the diagonal terms from the regularization, 
        and aims to minimize the pairwise cosine similarity between filters 
        without constraining their norm.

            beta * (||W^T W * (1 - I)||_F)^2

        where 
            || M ||_F = sqrt( sum_ij {| M_ij |} ) 

        is the Frobenius norm.
        """
        w_shape = tf.shape(w)
        bs, h, w, ch = w_shape[0], w_shape[1], w_shape[2], w_shape[-1]

        w = tf.reshape(w, [bs, h*w, ch])
        wtw = tf.matmul(w, w, transpose_a=True)
        ones = tf.ones_like(wtw)

        res = wtw * ones - wtw
        norm_squared = tf.reduce_sum(tf.abs(res))
        return self.beta * norm_squared

    def get_config(self):
        return {'beta': self.beta}


# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
class SpectralNormalization(tf.keras.layers.Wrapper):
    """Performs spectral normalization on weights.
    This wrapper controls the Lipschitz constant of the layer by
    constraining its spectral norm, which can stabilize the training of GANs.
    See [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).
    ```python
    net = SpectralNormalization(
        tf.keras.layers.Conv2D(2, 2, activation="relu"),
        input_shape=(32, 32, 3))(x)
    net = SpectralNormalization(
        tf.keras.layers.Conv2D(16, 5, activation="relu"))(net)
    net = SpectralNormalization(
        tf.keras.layers.Dense(120, activation="relu"))(net)
    net = SpectralNormalization(
        tf.keras.layers.Dense(n_classes))(net)
    ```
    Arguments:
      layer: A `tf.keras.layers.Layer` instance that
        has either `kernel` or `embeddings` attribute.
      power_iterations: `int`, the number of iterations during normalization.
    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not has `kernel` or `embeddings` attribute.
    """

    def __init__(self, layer: tf.keras.layers, power_iterations: int = 1, **kwargs):
        super().__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations
        self._initialized = False

    def build(self, input_shape):
        """Build `Layer`"""
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(
            shape=[None] + input_shape[1:])

        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor "
                "'embeddings'".format(type(self.layer).__name__)
            )

        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="sn_u",
            dtype=self.w.dtype,
        )

    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            self.normalize_weights()

        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    @tf.function
    def normalize_weights(self):
        """Generate spectral normalized weights.
        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
                u = tf.math.l2_normalize(tf.matmul(v, w))

            sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

            self.w.assign(self.w / sigma)
            self.u.assign(u)

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}


class ConditionalBatchNormalization(layers.Layer):
    def __init__(self, num_channels):
        """Layer for conditional batch normalization (normalization for GAN with label information)

        Args:
            num_channels (int): number of incoming channels
        """
        super().__init__()

        self.bn = layers.BatchNormalization(
            axis=-1, center=False, epsilon=1e-4)
        self.dense = layers.Dense(num_channels * 2)

    def call(self, x, y, training=True):
        """Performs a conditional normalization pass on the incoming data.

        Args:
            x (tf.Tensor): batch of feature maps
            y (tf.Tensor): batch of labels
            training (bool, optional): Defaults to True.

        Returns:
            tf.Tensor: batch normalized feature maps
        """
        x_shape = tf.shape(x)
        bs = x_shape[0]
        ch = x_shape[-1]

        emb = self.dense(y, training=training)
        bn = self.bn(x, training=training)
        gamma, beta = tf.split(emb, 2, axis=-1)

        gamma = tf.reshape(gamma, [bs, 1, 1, ch])
        beta = tf.reshape(beta, [bs, 1, 1, ch])

        bn = gamma * bn + beta

        return bn


class FullyConnected(layers.Layer):
    def __init__(self, units, sn=True):
        """Dense layer with spectral normalization

        Args:
            units (int): number of neurons
            sn (bool, optional): spectral normalization. Defaults to True.
        """
        super().__init__()
        self.units = units
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        if sn:
            self.fc = SpectralNormalization(layers.Dense(
                self.units,
                kernel_initializer=initializer,
            ))
        else:
            self.fc = layers.Dense(
                self.units,
                kernel_initializer=initializer,
            )

    def call(self, x, training=True):
        o = self.fc(x, training=training)
        return o


class Conv(layers.Layer):
    def __init__(self, filters, kernel=3, stride=1,
                 use_bias=True, sn=True, use_orth_init=True,
                 regularizer=None):
        """Conv2D layer with spectral normalization

        Args:
            filters (int): number of filters
            kernel (int, optional): kernel size. Defaults to 3.
            stride (int, optional): stride dimension. Defaults to 1.
            use_bias (bool, optional): learn bias vector. Defaults to True.
            sn (bool, optional): spectral normalization. Defaults to True.
            use_orth_init (bool, optional): use orthogonal initialization. Defaults to True.
            regularizer (tf.keras.regularizers.Regularizer, optional): regularizer. Defaults to None.
        """

        super().__init__()

        if use_orth_init:
            initializer = tf.keras.initializers.Orthogonal()
        else:
            initializer = tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02)

        if sn:
            self.conv = SpectralNormalization(layers.Conv2D(
                filters, kernel, stride,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                use_bias=use_bias,
                data_format='channels_last'
            ))
        else:
            self.conv = layers.Conv2D(
                filters, kernel, stride,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                use_bias=use_bias,
                data_format='channels_last'
            )

    def call(self, x, training=True):
        o = self.conv(x, training=training)
        return o


class Deconv(layers.Layer):
    def __init__(self, filters, kernel=3, stride=1,
                 use_bias=True, sn=True, use_orth_init=True,
                 regularizer=None):
        """Conv2DTranspose layer with spectral normalization

        Args:
            filters (int): number of filters
            kernel (int, optional): kernel size. Defaults to 3.
            stride (int, optional): stride dimension. Defaults to 1.
            use_bias (bool, optional): learn bias vector. Defaults to True.
            sn (bool, optional): spectral normalization. Defaults to True.
            use_orth_init (bool, optional): use orthogonal initialization. Defaults to True.
            regularizer (tf.keras.regularizers.Regularizer, optional): regularizer. Defaults to None.
        """
        super().__init__()

        if use_orth_init:
            initializer = tf.keras.initializers.Orthogonal()
        else:
            initializer = tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02)

        if sn:
            self.deconv = SpectralNormalization(layers.Conv2DTranspose(
                filters, kernel, stride,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                use_bias=use_bias,
                data_format='channels_last'
            ))
        else:
            self.deconv = layers.Conv2D(
                filters, kernel, stride,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                use_bias=use_bias,
                data_format='channels_last'
            )

    def call(self, x, training=True):
        o = self.deconv(x, training=training)
        return o


class ResBlock(layers.Layer):
    def __init__(self, in_channels, out_channels,
                 use_bias=True, sn=True,
                 regularizer=None):
        """Residual block that does not change feature map resolution

        Args:
            in_channels (int): number of incoming channels
            out_channels (int): number of outgoing channels
            use_bias (bool, optional): learn bias vector. Defaults to True.
            sn (bool, optional): use spectral normalization. Defaults to True.
            regularizer (tf.keras.regularizers.Regularizer, optional): regularizer. Defaults to None.
        """
        super().__init__()

        self.conv1 = Conv(in_channels, kernel=3, use_bias=use_bias, sn=sn,
                          regularizer=regularizer)

        self.conv2 = Conv(out_channels, kernel=3, use_bias=use_bias, sn=sn,
                          regularizer=regularizer)

    def residual(self, x, training=True):
        h = tf.nn.leaky_relu(x)
        h = self.conv1(h, training=training)

        h = tf.nn.leaky_relu(h)
        o = self.conv2(h, training=training)

        return o

    def shortcut(self, x, training=True):
        return x

    def call(self, x, training=True):
        residual = self.residual(x)
        shortcut = self.shortcut(x)

        return residual + shortcut


class ResBlockUp(layers.Layer):
    def __init__(self, in_channels, out_channels,
                 use_bias=True, sn=True,
                 regularizer=None):
        """Residual block that upscales the incoming feature map

        Args:
            in_channels (int): number of incoming channels
            out_channels (int): number of outgoing channels
            use_bias (bool, optional): learn bias vector. Defaults to True.
            sn (bool, optional): use spectral normalization. Defaults to True.
            regularizer (tf.keras.regularizers.Regularizer, optional): regularizer. Defaults to None.
        """
        super().__init__()

        self.cbn1 = ConditionalBatchNormalization(in_channels)
        self.deconv1 = Deconv(in_channels, kernel=3, stride=2,
                              use_bias=use_bias, sn=sn,
                              regularizer=regularizer)

        self.cbn2 = ConditionalBatchNormalization(in_channels)
        self.deconv2 = Deconv(out_channels, kernel=3, stride=1,
                              use_bias=use_bias, sn=sn,
                              regularizer=regularizer)

        self.deconv_sc = Deconv(out_channels, kernel=1, stride=2,
                                use_bias=use_bias, sn=sn,
                                regularizer=regularizer)

    def residual(self, x, y, training=True):
        h = self.cbn1(x, y, training=training)
        h = tf.nn.leaky_relu(h)
        h = self.deconv1(h, training=training)

        h = self.cbn2(h, y, training=training)
        h = tf.nn.leaky_relu(h)
        o = self.deconv2(h, training=training)

        return o

    def shortcut(self, x, training=True):
        o = self.deconv_sc(x, training=training)
        return o

    def call(self, x, y, training=True):
        residual = self.residual(x, y)
        shortcut = self.shortcut(x)

        return residual + shortcut


class ResBlockDown(layers.Layer):
    def __init__(self, in_channels, out_channels,
                 use_bias=True, sn=True,
                 regularizer=None):
        """Residual block that downscales the incoming feature map

        Args:
            in_channels (int): number of incoming channels
            out_channels (int): number of outgoing channels
            use_bias (bool, optional): learn bias vector. Defaults to True.
            sn (bool, optional): use spectral normalization. Defaults to True.
            regularizer (tf.keras.regularizers.Regularizer, optional): regularizer. Defaults to None.
        """
        super().__init__()

        self.conv1 = Conv(out_channels, kernel=3, stride=1,
                          use_bias=use_bias, sn=sn,
                          regularizer=regularizer)

        self.conv2 = Conv(out_channels, kernel=3, stride=2,
                          use_bias=use_bias, sn=sn,
                          regularizer=regularizer)

        self.conv_sc = Conv(out_channels, kernel=1, stride=2,
                            use_bias=use_bias, sn=sn,
                            regularizer=regularizer)

    def residual(self, x, training=True):
        h = tf.nn.leaky_relu(x)
        h = self.conv1(h, training=training)

        h = tf.nn.leaky_relu(h)
        o = self.conv2(h, training=training)

        return o

    def shortcut(self, x, training=True):
        o = self.conv_sc(x, training=training)
        return o

    def call(self, x, training=True):
        residual = self.residual(x)
        shortcut = self.shortcut(x)

        return residual + shortcut


class SelfAttention(layers.Layer):
    def __init__(
        self, channels, sn=True
    ):
        """Self-attention layer with spectral normalization

        Args:
            channels (int): number of incoming channels
            sn (bool, optional): use spectral normalization. Defaults to True.
        """
        super().__init__()
        self.channels = channels

        self.gamma = self.add_weight(
            name="gamma",
            shape=(),
            initializer=tf.initializers.constant(0.0),
            trainable=True,
        )

        self.conv_query = Conv(channels, kernel=1, use_bias=False, sn=sn)
        self.conv_key = Conv(channels, kernel=1, use_bias=False, sn=sn)
        self.conv_value = Conv(channels, kernel=1, use_bias=False, sn=sn)
        self.conv_attn = Conv(channels, kernel=1, use_bias=False, sn=sn)

    def call(self, x, training=True):
        input_shape = tf.shape(x)

        bs = input_shape[0]
        h = input_shape[1]
        w = input_shape[2]
        ch = self.channels

        # Computing key, query, value
        query = self.conv_query(x, training=training)
        key = self.conv_key(x, training=training)
        value = self.conv_value(x, training=training)

        # Flattening the "first" (after batch) dimension
        # (batch_size, N, channels)
        query = tf.reshape(query, [bs, h*w, ch])
        key = tf.reshape(key, [bs, h*w, ch])
        value = tf.reshape(value, [bs, h*w, ch])

        # [bs, ch/8, N]
        query = tf.transpose(query, [0, 2, 1])

        # Computing attention map
        attn = tf.matmul(key, query)  # [bs, N, ch] * [bs, ch, N] = [bs, N, N]
        attn = tf.nn.softmax(attn)   # [bs, N, N]

        # Computing self-attention feature maps
        attn = tf.matmul(attn, value)  # [bs, N, N] * [bs, N, ch] = [bs, N, ch]
        attn = tf.reshape(attn, [bs, h, w, ch])
        attn = self.conv_attn(attn, training=training)

        # Final residual (learnable) connection
        o = self.gamma * attn + x
        return o
