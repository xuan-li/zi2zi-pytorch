#!/bin/bash

FONT_FOLDER=$1

if [ ! -d "pairs" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir "pairs"
fi

i=0
for file in $FONT_FOLDER/*ttf; do
    echo "$(basename "$file")"
    if [ "$(basename "$file")" = "source.ttf" ];then
        continue
    else
        python font2img.py --src_font=$FONT_FOLDER/source.ttf --dst_font=$file --sample_dir="pairs" --label=$i
        i=$(($i+1))
    fi
done
echo "Totally $i fonts!"

python package.py --dir=pairs
rm -rf pairs