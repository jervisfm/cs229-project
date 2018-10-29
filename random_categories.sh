#!/bin/bash

# This is a script to randomly select categories to use for quickdraw dataset.

# It takes one argument which is the number of categories to generate.

NUM_CATEGORIES=$1
if [ -z "$NUM_CATEGORIES" ] 
then
    # Empty so use default num value.
    echo "Using default number of categories 10."
    NUM_CATEGORIES="10"
fi

# OS X unfortunately does not ship with shuf unix utility. So we manually string
# together various tools to randomly select categories
# - Generate a random number for each of the categories
# - Paste this random number infront of the line. 
# - Sort the lines which essentially sorts the random numbers
# - Take the first X of them to get the random selection.
# - Finally cut the path to just get category name.
jot -r "$(wc -l categories.txt)" 1 | paste - categories.txt | sort -n | cut -f 2- | head -n $NUM_CATEGORIES | cut -c 42- | rev | cut -c 5- | rev
