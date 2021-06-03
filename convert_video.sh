#!/bin/bash
VIDEO_RAW_FILES=/data/Projects/data/breaking_bad/raw/*
VIDEO_FOLDER=/data/Projects/data/breaking_bad/256x256/
for f in $VIDEO_RAW_FILES
do
  ff=${f##*/}
  fff=${ff%.*}
  echo "Processing $ff file, save to $VIDEO_FOLDER$ff"
#  echo "ffmpeg -i $f -vf scale=256:256 $VIDEO_FOLDER$fff.mp4"
  ffmpeg -i $f -vf scale=256:256 $VIDEO_FOLDER$fff.mp4
done