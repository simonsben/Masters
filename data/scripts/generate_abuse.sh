#!/bin/bash

cd ../prepared_data/

FILENAME="abusive_data"
echo "Generating $FILENAME"

cat "hate_speech_dataset.csv" > "$FILENAME.csv"
tail -n +2 "kaggle.csv" >> "$FILENAME.csv"
tail -n +2 "insults.csv" >> "$FILENAME.csv"

echo "Standard complete."

cat "hate_speech_dataset_partial.csv" > "${FILENAME}_partial.csv"
tail -n +2 "kaggle_partial.csv" >> "${FILENAME}_partial.csv"
tail -n +2 "insults_partial.csv" >> "${FILENAME}_partial.csv"

echo "Partial complete."
