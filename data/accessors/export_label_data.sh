#!/bin/bash

DATABASE="database_may_1.sdb"

# Move to dataset directory
cd "../datasets/"

LABEL_DIR="data_labelling/"
if [ ! -d $LABEL_DIR ]; then
  mkdir $LABEL_DIR
fi

# Get labelling contexts
sqlite3 -header -csv $DATABASE "SELECT * FROM context;" > $LABEL_DIR"label_contexts.csv"

# Get labels
sqlite3 -header -csv $DATABASE "SELECT * FROM label;" > $LABEL_DIR"labels.csv"
