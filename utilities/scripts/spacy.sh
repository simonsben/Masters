#!/bin/bash

# TODO double check this works
# Enter virtual environment
source ../../venv/Scripts/activate

# Load spacy model
python3 -m spacy download en_core_web_md
