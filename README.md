# Automatic Modulation Classification with Deep Neural Networks

## Download Dataset

The dataset can be downloaded from https://www.deepsig.ai/datasets.

We use the RADIOML 2018.01A dataset.

## Setup

To run our code as a package, we provide a `setup.py` file.  To run the setup, inside this directory, run:

```bash
python setup.py develop
```

## Prepare Dataset

Our code assumes the data to be in TFRecord format; however, the dataset provided by DeepSig is in hdf5 format.  To convert the dataset using our train/test split, run the script `process_data.py`.  For example:

```bash
process_data.py --src_dataset_path="./dataset/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5" --dest_dataset_path="./dataset/GOLD_XYZ_OSC_tfrecord"
```

To perform the same operation with less code and put the data directly into `tensorflow_datasets` ([tfds](https://www.tensorflow.org/datasets/overview)) format, please see another repository of ours:
https://github.com/caharper/smart-tfrecord-writer/tree/main

With the following example:
https://github.com/caharper/smart-tfrecord-writer/blob/main/examples/radioml/radioml_to_tfrecord.py

We will use this code in future work as it is more flexible and easier to use.  Particularly, loading the data becomes much easier with tfds using `tfds.load()`.

## Run Code

To run all experiments using `Bocas`, run the following inside the `./experiments/radioml2018.01a/` directory:

```bash
python3 -m bocas.launch run.py --task run.py --config configs/sweep-models.py
```

## Citation

If you make use of this work or use our dataset split, please cite our work:

```bibtex
@Article{electronics12183962,
  AUTHOR = {Harper, Clayton A. and Thornton, Mitchell A. and Larson, Eric C.},
  TITLE = {Automatic Modulation Classification with Deep Neural Networks},
  JOURNAL = {Electronics},
  VOLUME = {12},
  YEAR = {2023},
  NUMBER = {18},
  ARTICLE-NUMBER = {3962},
  URL = {https://www.mdpi.com/2079-9292/12/18/3962},
  ISSN = {2079-9292},
  DOI = {10.3390/electronics12183962}
}
```

The link to our paper can be found here:
  https://www.mdpi.com/2079-9292/12/18/3962

The preprint can be found here:
https://arxiv.org/abs/2301.11773
