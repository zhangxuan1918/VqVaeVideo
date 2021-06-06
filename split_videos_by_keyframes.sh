#!/bin/bash

show=
VIDEO_FILES=/data/Projects/data/$show/256x256/*
SAVE_FOLDER=/data/Projects/data/$show/video_clips/256x256/
for f in $VIDEO_FILES
do
  ff=${f##*/}
  fff=${ff%.*}
  echo "Processing $f file, save to $SAVE_FOLDER$fff"
#  echo "ffmpeg -i $f -acodec copy -f segment -vcodec copy -reset_timestamps 1 -map 0 $SAVE_FOLDER$fff-%d.mp4"
  ffmpeg -i $f -acodec copy -f segment -vcodec copy -reset_timestamps 1 -map 0 $SAVE_FOLDER$fff-%d.mp4
done