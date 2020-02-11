#!/bin/bash

if [ ! -f "../prepared_data/abusive_data.csv" ]; then
  echo "Abusive data not generated, running first."
  bash "generate_abuse.sh"
fi

cd ../prepared_data/

FILENAME="abusive_intent"
echo "Generating $FILENAME"

cat "abusive_data.csv" > "$FILENAME.csv"
tail -n +2 "storm-front.csv" >> "$FILENAME.csv"

echo "Standard complete."

cat "abusive_data_partial.csv" > "${FILENAME}_partial.csv"
tail -n +2 "storm-front_partial.csv" >> "${FILENAME}_partial.csv"

echo "Partial complete."
