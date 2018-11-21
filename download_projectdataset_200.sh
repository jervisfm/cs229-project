#!/bin/bash

# This is a script to download Google Quick Draw dataset.

echo "Downloading quickdraw dataset from Google. This will take a moment ..."
DEST_DIR="data/numpy_bitmap_200"
mkdir -p $DEST_DIR


CATEGORIES="beach
The Eiffel Tower
clock
elephant
eyeglasses
nose
roller coaster
suitcase
wine bottle
ice cream
see saw
umbrella
anvil
bridge
crown
bat
foot
speedboat
stethoscope
hamburger
dragon
keyboard
remote control
belt
flower
garden hose
piano
blackberry
candle
coffee cup
knife
aircraft carrier
arm
barn
campfire
cruise ship
microwave
mug
hammer
leg
toaster
waterslide
ant
bus
crayon
binoculars
helmet
t-shirt
The Mona Lisa
backpack
basket
hurricane
ocean
screwdriver
donut
megaphone
tornado
carrot
duck
finger
hourglass
pineapple
potato
parrot
sea turtle
camera
mermaid
toothpaste
bandage
house
school bus
shoe
string bean
diamond
hexagon
eraser
knee
lighthouse
spreadsheet
bench
bulldozer
cup
lighter
pants
rain
rake
syringe
matches
cow
light bulb
lightning
rainbow
ear
hat
onion
couch
moustache
rabbit
snail
television
train
book
microphone
cat
ceiling fan
jacket
panda
shovel
axe
oven
snowflake
angel
chair
violin
fish
helicopter
necklace
baseball bat
lollipop
sailboat
saxophone
shorts
firetruck
guitar
computer
dishwasher
hedgehog
traffic light
penguin
rifle
birthday cake
crab
door
stitches
lobster
saw
yoga
drums
lipstick
bush
sandwich
stove
asparagus
cooler
snowman
bucket
calendar
circle
alarm clock
bathtub
castle
passport
triangle
windmill
zebra
canoe
tiger
fan
flying saucer
police car
radio
sock
tennis racquet
broccoli
map
paint can
palm tree
pickup truck
shark
banana
crocodile
toilet
butterfly
lantern
pencil
wheel
horse
watermelon
basketball
fence
sun
grapes
paper clip
peas
stereo
washing machine
wristwatch
frying pan
garden
mushroom
bird
owl
tent
cactus
fireplace
hot tub
scorpion
streetlight
hot air balloon
snake"

for category in $CATEGORIES
do
    gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/$category.npy "$DEST_DIR"
done

echo "Downloading done."
