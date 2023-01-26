import tensorflow as tf


class GlobalVariancePooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = False

    def call(self, inputs):
        return tf.math.reduce_variance(inputs, axis=1)
