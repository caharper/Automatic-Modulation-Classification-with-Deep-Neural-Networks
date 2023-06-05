import bocas
import ml_collections
import tensorflow as tf


def get_config():
    config = ml_collections.ConfigDict()

    # Dataset info
    config.data_path = "../../../dataset/GOLD_XYZ_OSC_tfrecord"

    # Model hyperparameters
    config.use_se = bocas.Sweep([True, False])
    config.use_act = bocas.Sweep([True, False])
    config.residual = bocas.Sweep([True, False])
    config.dilate = bocas.Sweep([True, False])
    config.self_attention = bocas.Sweep([True, False])

    # Training hyperparameters
    config.batch_size = 128
    config.epochs = 1000
    config.verbose = 2

    return config
