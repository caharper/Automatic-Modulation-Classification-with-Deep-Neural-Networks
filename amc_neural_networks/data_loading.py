import tensorflow as tf
import os
from glob import glob


@tf.function
def get_tf_batch(elements, with_snr=False):
    data = {
        "observation": tf.io.FixedLenFeature([1024, 2], dtype=tf.float32),
        "label": tf.io.FixedLenFeature([], dtype=tf.int64),
        "snr": tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    content = tf.io.parse_example(elements, data)

    features = content["observation"]
    labels = content["label"]

    if with_snr:
        snrs = content["snr"]
        return features, labels, snrs

    return features, labels


def load_tfrecords(tfrecords_dir="tfrecords", split="train"):

    # List all *.tfrecord files for the selected split
    pattern = os.path.join(tfrecords_dir, "{}*.tfrecord".format(split))
    files_ds = tf.data.Dataset.list_files(pattern)

    return files_ds


def get_n_tfrecord_files(dataset_dir, split):
    pattern = os.path.join(dataset_dir, "{}*.tfrecord".format(split))
    return len(glob(pattern))
