import numpy as np
import os
import tensorflow as tf
import h5py
from sklearn.utils import shuffle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--src_dataset_path", type=str, help="Path to the source dataset",
)

parser.add_argument(
    "--dest_dataset_path", type=str, help="Path to save the generated TFRecord dataset",
)

parser.add_argument(
    "--train_indexes_path",
    type=str,
    default="./dataset/train_indexes.csv",
    help="Path to the training indexes",
)

parser.add_argument(
    "--test_indexes_path",
    type=str,
    default="./dataset/test_indexes.csv",
    help="Path to the test indexes",
)

parser.add_argument(
    "--n_per_shard",
    type=int,
    default=2_000,
    help="Approximate number of examples per shard",
)

parser.add_argument(
    "--shuffle_data",
    type=bool,
    default=True,
    help="Whether to shuffle the train and test indexes",
)

args = parser.parse_args()


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class Shard:
    def __init__(self, shard_indexes, split, shard_idx, split_total_shards):
        self.shard_indexes = shard_indexes

        self.shard_path = self._get_shard_path(
            split, shard_idx, split_total_shards, len(str(split_total_shards))
        )

    def _get_shard_path(self, split, shard_idx, total_shards, width_display):
        return f"{split}-{shard_idx:{0}{width_display}}-{total_shards}.tfrecord"


class TFRecordWriter:
    def __init__(self, dest_dataset_path, src_dataset_path, log_interval=0.2):
        self.dest_dataset_path = dest_dataset_path
        self.src_dataset_path = src_dataset_path
        if os.path.exists(dest_dataset_path):
            print(f"{dest_dataset_path} already exists")
        else:
            os.mkdir(dest_dataset_path)
        self.log_interval = log_interval

    def _create_tf_example(self, observation, label, snr):

        data_dict = {
            "observation": _float_feature(observation.flatten().tolist()),
            "label": _int64_feature(label),
            "snr": _int64_feature(snr),
        }

        # create an Example
        out = tf.train.Example(features=tf.train.Features(feature=data_dict))

        return out

    def _write_shard(self, shard_path, shard_data, shard_labels, shard_snrs):

        with tf.io.TFRecordWriter(
            os.path.join(self.dest_dataset_path, shard_path)
        ) as out:
            for observation, label, snr in tqdm(
                zip(shard_data, shard_labels, shard_snrs),
                leave=True,
                position=0,
                total=len(shard_labels),
            ):
                example = self._create_tf_example(observation, label, snr)
                out.write(example.SerializeToString())

    def _convert_shards(self, shards):

        total_shards = len(shards)
        log_step = int(total_shards * self.log_interval)

        # Avoid division by 0
        if log_step == 0:
            log_step = 1

        with h5py.File(self.src_dataset_path, "r") as f:

            for i, shard in enumerate(shards, 1):
                shard_data = []
                shard_labels = []
                shard_snrs = []

                for idx in shard.shard_indexes:
                    shard_data.append(f["X"][idx])
                    shard_labels.append(f["Y"][idx])
                    shard_snrs.append(f["Z"][idx])

                shard_data = np.asarray(shard_data)
                shard_labels = np.argmax(np.asarray(shard_labels), axis=1)
                shard_snrs = np.squeeze(np.asarray(shard_snrs))

                self._write_shard(
                    shard.shard_path, shard_data, shard_labels, shard_snrs
                )

                if i % log_step == 0:
                    print(f"\t{i}/{total_shards} shards written")

    def _convert_to_shard(self, shard_list, split):
        total_shards = len(shard_list)
        return [
            Shard(shard_indexes, split, i, total_shards)
            for i, shard_indexes in enumerate(shard_list)
        ]

    def convert(
        self, train_shard_list=None, test_shard_list=None, validation_shard_list=None
    ):

        shards = [train_shard_list, test_shard_list, validation_shard_list]
        shard_names = ["train", "test", "validation"]

        for shard_list, split in zip(shards, shard_names):
            if shard_list is not None:
                processed_shards = self._convert_to_shard(shard_list, split)
                print(f"Starting {split} set writing...")
                self._convert_shards(processed_shards)


if __name__ == "__main__":
    train_indexes = np.genfromtxt(args.train_indexes_path, delimiter=",", dtype=np.int)
    test_indexes = np.genfromtxt(args.test_indexes_path, delimiter=",", dtype=np.int)

    # Shuffle indexes to avoid spurious info
    if args.shuffle_data:
        train_indexes = shuffle(train_indexes)
        test_indexes = shuffle(test_indexes)

    # Get how many observations per shard
    n_train_shards = np.ceil(len(train_indexes) / args.n_per_shard)
    n_test_shards = np.ceil(len(test_indexes) / args.n_per_shard)

    # Split indexes into list of indexes for n_shards
    train_shard_indexes = np.array_split(train_indexes, n_train_shards)
    test_shard_indexes = np.array_split(test_indexes, n_test_shards)

    print(
        f"There are {len(train_shard_indexes) + len(test_shard_indexes)} total shards."
    )

    writer = TFRecordWriter(
        args.dest_dataset_path,
        src_dataset_path=args.src_dataset_path,
        log_interval=0.05,
    )

    writer.convert(
        train_shard_list=train_shard_indexes, test_shard_list=test_shard_indexes
    )
