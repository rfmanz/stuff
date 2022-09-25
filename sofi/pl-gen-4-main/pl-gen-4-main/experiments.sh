#!/bin/bash

FILES="./configs/*"
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  python main.py -c $f
done