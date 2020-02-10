#!/bin/bash

cd ../prepared_data/

$FILENAME = "abusive_intent"

cat "hannah_dataset.csv" > "$FILENAME.csv"
tail -n +2 "storm-front.csv" >> "$FILENAME.csv"

cat "hannah_dataset_partial.csv" > "${FILENAME}_partial.csv"
tail -n +2 "storm-front_partial.csv" >> "${FILENAME}_partial.csv"
