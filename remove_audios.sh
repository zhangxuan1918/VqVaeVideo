#!/bin/bash

show=westworld
VIDEO_FILES=/data/Projects/data/$show/video_clips/256x256/*
SAVE_FOLDER=/data/Projects/data/$show/video_clips/256x256-2/
for f in $VIDEO_FILES
do
  ff=${f##*/}
  fff=${ff%.*}
#  echo "Processing $f file, save to $SAVE_FOLDER$fff"
#  echo "ffmpeg -i $f -c copy -an $SAVE_FOLDER$ff"
  ffmpeg -i $f -c copy -an $SAVE_FOLDER$ff
done