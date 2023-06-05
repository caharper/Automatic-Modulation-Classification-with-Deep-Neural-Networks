import tensorflow as tf


def se_block(x, block_num, residual=False):
    *_, n_filters = tf.keras.backend.int_shape(x)
    se_shape = (1, n_filters)

    se = tf.keras.layers.GlobalAveragePooling1D(name=f"se_block_{block_num}_stat")(x)
    se = tf.keras.layers.Reshape(se_shape, name=f"se_block_{block_num}_reshape")(se)

    se = tf.keras.layers.Dense(
        n_filters // 2,
        activation=None,
        kernel_initializer="he_normal",
        use_bias=False,
        name=f"se_block_{block_num}_dense_1",
    )(se)

    se = tf.keras.layers.ReLU()(se)
    se = tf.keras.layers.Dense(
        n_filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
        name=f"se_block_{block_num}_dense_2",
    )(se)

    scale = tf.keras.layers.multiply([x, se], name=f"se_block_{block_num}_mult")

    if residual:
        scale = x + scale
    return scale
