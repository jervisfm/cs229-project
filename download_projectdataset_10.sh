#!/bin/bash

# This is a script to download Google Quick Draw dataset.

#Set the field separator to new line
IFS=$'\n'

echo "Downloading quickdraw dataset from Google. This will take a moment ..."
DEST_DIR="data/numpy_bitmap_10"
mkdir -p $DEST_DIR

CATEGORIES="laptop
rain
hockey stick
wristwatch
nose
pond
The Mona Lisa
banana
panda
paint can"

for category in $CATEGORIES
do
    gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/$category.npy $DEST_DIR
done

echo "Downloading done."
