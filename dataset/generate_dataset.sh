#!/bin/bash

FONT_FOLDER=$1
TARGET_FOLDER=$2

if [ ! -d $TARGET_FOLDER ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir $TARGET_FOLDER
fi

i=1
for file in $FONT_FOLDER/*ttf; do
    if [ "$(basename "$file")" = "source.tff" ];then
        continue
    else
        python font2img.py --src_font=$FONT_FOLDER/source.ttf --dst_font=$file --sample_dir=$TARGET_FOLDER --label=$i
        i=$(($i+1))
    fi
done