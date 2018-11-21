#!/bin/bash

# This is a script to download Google Quick Draw dataset.

echo "Downloading quickdraw dataset from Google. This will take a moment ..."
mkdir -p data/numpy_bitmap_fulldataset
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy data/numpy_bitmap_fulldataset
echo "Downloading done."
