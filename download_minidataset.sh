#!/bin/bash

# This is a script to download Google Quick Draw dataset.

echo "Downloading quickdraw dataset from Google. This will take a moment ..."
mkdir -p data/numpy_bitmap


CATEGORIES="eraser
belt
banana"

for category in $CATEGORIES
do
    gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/$category.npy data/numpy_bitmap
done

echo "Downloading done."
