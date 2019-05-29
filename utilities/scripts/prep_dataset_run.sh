#!/bin/bash

# Check for dataset name argument
if [ $# -eq 0 ]; then
    echo "No dataset (name) supplied, must be given as first argument"
    exit 1
fi


# Define sub directories
declare -a DIRS=('models' 'processed_data' '../figures')
declare -a SUB_DIRS=('derived' 'emotion' 'lexicon')

# Move to data directory
cd ../../data/
REF_DIR=$PWD

# Make model folders
for DIR in "${DIRS[@]}"; do
	cd "$REF_DIR/$DIR"

	if [ ! -d $1 ]; then
		mkdir $1
		cd $1

		for SUB_DIR in "${SUB_DIRS[@]}"; do
			mkdir ${SUB_DIR}
		done
	fi
done
