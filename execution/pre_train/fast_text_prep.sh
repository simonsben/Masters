#!/bin/bash

if [ -z "$1" ]; then
        echo "No data name supplied"
        exit 1;
fi

cd "../../data/prepared_data/"

tail -n +2 "$1.csv" | grep -Eo "[a-zA-Z ]+" > "../../../fastText/data/$1.csv"
