#!/bin/bash
# run with bash not sh in the OCRapus train data dir
#convert images and move to tesseract data directory
for file in ./*
do
	if [[ "$file" == *".png"* ]]; then
		new_name=$(echo $file | sed -e "s/.png/.tif/g; s/.\///g")
		echo $new_name
		new_name='../../../tesstrain/data/ground-truth/'$new_name
		convert "$file" "$new_name"
	fi
done
