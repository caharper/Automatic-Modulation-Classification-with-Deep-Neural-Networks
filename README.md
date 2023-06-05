# Download Dataset

The dataset can be downloaded from https://www.deepsig.ai/datasets.

We use the RADIOML 2018.01A dataset.

# Setup

To run our cod as a package, we provide a `setup.py` file.  To run the setup, inside this directory, run:

```bash
python setup.py develop
```

# Prepare Dataset

Our code assumes the data to be in TFRecord format; however, the dataset provided by DeepSig is in hdf5 format.  To convert the dataset using our train/test split, run the script `process_data.py`.  For example:

```
    process_data.py --src_dataset_path="./dataset/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5" --dest_dataset_path="./dataset/GOLD_XYZ_OSC_tfrecord"
```

# Run Code

To run all experiments using `Bocas`, run the following inside the `./experiments/radioml2018.01a/` directory:

```bash
python3 -m bocas.launch run.py --task run.py --config configs/sweep-models.py
```


# Citation

If you make use of this work or use our dataset split, please cite our work:

```
@article{harper2023automatic,
  title={Automatic Modulation Classification with Deep Neural Networks},
  author={Harper, Clayton and Thornton, Mitchell and Larson, Eric},
  journal={arXiv preprint arXiv:2301.11773},
  year={2023}
}
```

The link to our paper can be found here: https://arxiv.org/abs/2301.11773