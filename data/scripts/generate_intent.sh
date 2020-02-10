#!/bin/bash

cd ../prepared_data/

$FILENAME = "abusive_data"

cat "hate_speech_dataset.csv" > "$FILENAME.csv"
tail -n +2 "insults.csv" >> "$FILENAME.csv"

cat "hate_speech_dataset_partial.csv" > "${FILENAME}_partial.csv"
tail -n +2 "insults_partial.csv" >> "${FILENAME}_partial.csv"
