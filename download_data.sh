#!/bin/bash

# This is a script to download Google Quick Draw dataset.

echo "Downloading quickdraw dataset from Google. This will take a moment ..."
mkdir -p data/numpy_bitmap
gsutil -m cp gs://quickdraw_dataset/full/simplified/*.ndjson .

gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.ndjson data/numpy_bitmap
echo "Downloading done."
