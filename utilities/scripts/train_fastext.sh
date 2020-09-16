#!/bin/bash

# NOTE: This script has to be relocated into the fasttext build folder and stands just to show the options used

./fasttext skipgram -input "data/$1" -output "data/$2" -dim 200 -thread 3 &
