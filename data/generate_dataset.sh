#!/bin/bash

FONT_FOLDER=$1
TARGET_FOLDER=$2
mkdir $TARGET_FOLDER

i=1
for file in $(find $FONT_FOLDER -name *.tff); 
do
    echo $file
    if ["$(basename "$file")"="source.tff"];
    then
        continue
    else
        a=$(($a+1))
    fi
done
