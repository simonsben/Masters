#!/bin/bash

if [ -z "$1" ]; then
        echo "No data name supplied"
        exit 1;
fi

cd "../../data/prepared_data/"

cat "$1.csv" | awk -F "," '{print $NF}' > "../../../fastText/data/$1.csv"
