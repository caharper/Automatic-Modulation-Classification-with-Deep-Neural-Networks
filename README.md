# Download Dataset

The dataset can be downloaded from https://www.deepsig.ai/datasets.

We use the RADIOML 2018.01A dataset.

# Prepare Dataset

Our code assumes the data to be in TFRecord format; however, the dataset provided by DeepSig is in hdf5 format.  To convert the dataset using our train/test split, run the script `process_data.py`.  For example:

```
    process_data.py --src_dataset_path="./dataset/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5" --dest_dataset_path="./dataset/GOLD_XYZ_OSC_tfrecord"
```

# Run Code

The `notebooks` directory contains and example notebook on how to train a network with our configuration.  By adjusting the parameters to `create_model()`, our experiments can be recreated.

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