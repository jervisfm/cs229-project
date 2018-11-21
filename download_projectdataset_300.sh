#!/bin/bash

# This is a script to download Google Quick Draw dataset.

#Set the field separator to new line
IFS=$'\n'

echo "Downloading quickdraw dataset from Google. This will take a moment ..."
DEST_DIR="data/numpy_bitmap_300"
mkdir -p $DEST_DIR


CATEGORIES="hand
peas
clock
cup
eye
sink
drums
sailboat
sock
cooler
bottlecap
crab
drill
frog
giraffe
owl
camouflage
circle
dragon
flower
headphones
hot tub
tree
diamond
line
mouse
pickup truck
feather
toaster
chandelier
bird
computer
aircraft carrier
firetruck
pond
teddy-bear
hat
remote control
trumpet
necklace
paper clip
rake
syringe
telephone
waterslide
bed
dolphin
palm tree
pencil
radio
scissors
triangle
bowtie
skyscraper
sleeping bag
rifle
fan
hourglass
house
stop sign
watermelon
cell phone
envelope
The Mona Lisa
camel
lipstick
map
swing set
bat
chair
keyboard
mosquito
passport
rain
donut
lantern
train
anvil
cloud
marker
suitcase
truck
flashlight
tennis racquet
dishwasher
bucket
compass
postcard
see saw
zigzag
clarinet
cow
monkey
t-shirt
television
rollerskates
book
hedgehog
hurricane
oven
sandwich
bicycle
helmet
moustache
saw
stereo
tiger
frying pan
washing machine
floor lamp
hammer
lobster
snowman
apple
moon
pear
airplane
shovel
vase
screwdriver
couch
ice cream
mouth
speedboat
door
hockey puck
motorbike
shoe
wine glass
beach
bush
diving board
hexagon
parachute
rabbit
whale
crayon
purse
crocodile
octopus
parrot
shark
cake
campfire
paintbrush
leg
pliers
rhinoceros
strawberry
umbrella
jail
smiley face
crown
dresser
lightning
snowflake
tent
fish
jacket
hot air balloon
brain
carrot
garden
harp
key
panda
popsicle
traffic light
blueberry
broom
cat
nail
skull
soccer ball
backpack
stairs
string bean
teapot
underwear
yoga
picture frame
ceiling fan
ear
knee
ladder
pig
swan
candle
guitar
butterfly
stitches
alarm clock
blackberry
angel
bathtub
roller coaster
shorts
stove
sword
ambulance
asparagus
flying saucer
hospital
toothbrush
bear
bread
lollipop
onion
helicopter
pants
toothpaste
bridge
lighter
toilet
van
The Eiffel Tower
castle
knife
birthday cake
cannon
church
face
horse
snorkel
bulldozer
fire hydrant
kangaroo
squiggle
wheel
finger
foot
saxophone
star
beard
fireplace
ocean
toe
bracelet
cactus
cookie
leaf
mushroom
rainbow
tractor
bandage
submarine
tornado
The Great Wall of China
animal migration
cello
hamburger
light bulb
violin
dumbbell
hot dog
fence
microphone
police car
pool
snake
stethoscope
banana
basket
microwave
bus
grapes
laptop
matches
tooth
basketball
bee
elephant
house plant
raccoon
broccoli
garden hose
megaphone
octagon
pillow
dog
paint can
peanut
piano
bench
canoe
flip flops
mermaid
spreadsheet
goatee
golf club
scorpion
square
baseball bat
duck
river
sheep
arm
binoculars
power outlet
spider"

for category in $CATEGORIES
do
    gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/$category.npy "$DEST_DIR"
done

echo "Downloading done."
