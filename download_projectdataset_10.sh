#!/bin/bash

# This is a script to download Google Quick Draw dataset.

#Set the field separator to new line
IFS=$'\n'

echo "Downloading quickdraw dataset from Google. This will take a moment ..."
mkdir -p data/numpy_bitmap

CATEGORIES="squirrel
paint can
stitches
lighter
panda
hockey stick
kangaroo
rain
sea turtle
aircraft carrier"

for category in $CATEGORIES
do
    gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/$category.npy data/numpy_bitmap
done

echo "Downloading done."
