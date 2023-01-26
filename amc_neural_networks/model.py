import tensorflow as tf
from .layers import GlobalVariancePooling1D


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


def create_model(
    use_se=True,
    use_act=True,
    residual=False,
    dilate=True,
    self_attention=False,
    input_shape=(1024, 2),
    n_classes=24,
):

    if dilate:
        dilation_rates = [1, 2, 3, 2, 2, 2, 1]
    else:
        dilation_rates = [1] * 7

    inputs = tf.keras.layers.Input(shape=input_shape, name="input")
    net = inputs

    net = tf.keras.layers.Conv1D(
        32,
        7,
        dilation_rate=dilation_rates[0],
        padding="same",
        activation="relu",
        name="conv_1",
    )(net)

    if use_se:
        net = se_block(net, "1", residual=residual)

    net = tf.keras.layers.Conv1D(
        48,
        5,
        dilation_rate=dilation_rates[1],
        padding="same",
        activation="relu",
        name="conv_2",
    )(net)

    if use_se:
        net = se_block(net, "2", residual=residual)

    net = tf.keras.layers.Conv1D(
        64,
        7,
        dilation_rate=dilation_rates[2],
        padding="same",
        activation="relu",
        name="conv_3",
    )(net)

    if use_se:
        net = se_block(net, "3", residual=residual)

    net = tf.keras.layers.Conv1D(
        72,
        5,
        dilation_rate=dilation_rates[3],
        padding="same",
        activation="relu",
        name="conv_4",
    )(net)

    if use_se:
        net = se_block(net, "4", residual=residual)

    net = tf.keras.layers.Conv1D(
        84,
        3,
        dilation_rate=dilation_rates[4],
        padding="same",
        activation="relu",
        name="conv_5",
    )(net)

    if use_se:
        net = se_block(net, "5", residual=residual)

    net = tf.keras.layers.Conv1D(
        96,
        3,
        dilation_rate=dilation_rates[5],
        padding="same",
        activation="relu",
        name="conv_6",
    )(net)

    if use_se:
        net = se_block(net, "6", residual=residual)

    if use_act:
        net = tf.keras.layers.Conv1D(
            108,
            3,
            dilation_rate=dilation_rates[6],
            padding="same",
            name="conv_7",
            activation="relu",
        )(net)
    else:
        net = tf.keras.layers.Conv1D(
            108, 3, dilation_rate=dilation_rates[6], padding="same", name="conv_7"
        )(net)

    if self_attention:
        net = tf.keras.layers.MultiHeadAttention(1, 108)(net, net)

    # X-Vector Pooling
    gap = tf.keras.layers.GlobalAveragePooling1D(name="gap")(net)
    gvp = GlobalVariancePooling1D(name="gvp")(net)

    net = tf.keras.layers.concatenate([gap, gvp], name="stat_pool")

    net = tf.keras.layers.Dense(128, name="dense_variant_1")(net)
    net = tf.keras.layers.Activation("selu", name="act_variant_1")(net)

    net = tf.keras.layers.Dense(128, name="dense_variant_2")(net)
    net = tf.keras.layers.Activation("selu", name="act_variant_2")(net)

    variant_output = tf.keras.layers.Dense(
        n_classes, activation="softmax", name="variant"
    )(net)

    return tf.keras.models.Model(inputs, variant_output)
