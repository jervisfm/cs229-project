#!/bin/bash

# This is a script to download Google Quick Draw dataset.

echo "Downloading quickdraw dataset from Google. This will take a moment ..."
DEST_DIR="data/numpy_bitmap_3"
mkdir -p $DEST_DIR


CATEGORIES="eraser
belt
banana"

for category in $CATEGORIES
do
    gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/$category.npy "$DEST_DIR"
done

echo "Downloading done."
