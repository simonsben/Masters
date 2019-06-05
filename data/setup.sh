#!/bin/bash

# Create folders
declare -a DIRS=('models' 'predictions' 'prepared_data' 'prepared_lexicon' 'processed_data' '../figures' 'lexicons/fast_text')

for DIR in "${DIRS[@]}"; do
    if [ ! -d $DIR ]; then
        mkdir ${DIR}
    fi
done

#cd datasets

cd lexicons
wget -O fast_text.zip https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
