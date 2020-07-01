#!/bin/bash

CORPUS_NAME="wikipedia_corpus"

cd "../datasets/"
if [ ! -d "$CORPUS_NAME/" ]; then
  echo "Creating corpus directory"
  mkdir "$CORPUS_NAME/"
fi

cd "$CORPUS_NAME"
FILE_PATH="$CORPUS_NAME.xml.bz2"

if [ ! -f "$FILE_PATH" ]; then
  wget -O "$FILE_PATH" "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
fi
