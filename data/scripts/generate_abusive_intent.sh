#!/bin/bash

cd ../prepared_data/

FILENAME="abusive_intent"
echo "Generating $FILENAME"

cat "hannah_data.csv" > "$FILENAME.csv"
tail -n +2 "storm-front.csv" >> "$FILENAME.csv"

echo "Standard complete."

cat "hannah_data_partial.csv" > "${FILENAME}_partial.csv"
tail -n +2 "storm-front_partial.csv" >> "${FILENAME}_partial.csv"

echo "Partial complete."
