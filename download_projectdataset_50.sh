#!/bin/bash

# This is a script to download Google Quick Draw dataset.

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
aircraft carrier
beard
chair
hammer
hedgehog
spreadsheet
camel
nose
peas
wristwatch
banana
belt
eraser
harp
shark
skull
face
castle
drill
lobster
snowflake
speedboat
bee
fence
floor lamp
hat
leaf
ambulance
pond
sweater
grapes
laptop
screwdriver
broccoli
finger
teddy-bear
The Mona Lisa"

for category in $CATEGORIES
do
    gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/$category.npy data/numpy_bitmap
done

echo "Downloading done."
