#!/bin/bash

# This is a script to download Google Quick Draw dataset.

echo "Downloading quickdraw dataset from Google. This will take a moment ..."
DEST_DIR="data/numpy_bitmap_100"
mkdir -p $DEST_DIR


CATEGORIES="microphone
cruise ship
watermelon
bridge
hot air balloon
mug
passport
carrot
map
ocean
washing machine
lantern
lighthouse
bread
frying pan
hockey stick
rainbow
bracelet
camera
dishwasher
lightning
violin
oven
diving board
foot
soccer ball
swan
angel
axe
belt
cell phone
envelope
saw
vase
stop sign
drill
grapes
hedgehog
nail
pool
flashlight
moustache
remote control
campfire
cello
broccoli
sailboat
scorpion
bicycle
harp
hat
trombone
blackberry
cup
garden hose
shorts
sweater
wristwatch
The Eiffel Tower
feather
pants
potato
cloud
firetruck
ice cream
pineapple
postcard
church
dumbbell
leg
rake
sheep
string bean
ant
headphones
airplane
arm
basketball
computer
elbow
kangaroo
ambulance
crocodile
sleeping bag
suitcase
table
parachute
popsicle
rollerskates
squirrel
apple
penguin
house plant
barn
lobster
toe
book
flamingo
laptop
toilet"

for category in $CATEGORIES
do
    gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/$category.npy "$DEST_DIR"
done

echo "Downloading done."
