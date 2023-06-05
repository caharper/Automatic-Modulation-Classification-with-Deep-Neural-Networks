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


def load(dataset_root_path, split, batch_size, shuffle_len=None, with_snr=False):
    ds = load_tfrecords(dataset_root_path, split=split)

    if shuffle_len is not None:
        ds = ds.shuffle(shuffle_len, reshuffle_each_iteration=True)

    ds = (
        ds.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .map(lambda x: get_tf_batch(x, with_snr=with_snr))
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds.prefetch(tf.data.AUTOTUNE)
