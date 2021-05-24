#!/bin/bash
VIDEO_FILES=/data/Projects/data/Doraemon/video_clips/*
IMAGE_FOLDER=/data/Projects/data/Doraemon/images/
for f in $VIDEO_FILES
do
  ff=${f##*/}
  fff=${ff%.*}
  echo "Processing $ff file, save to $IMAGE_FOLDER$fff"
#  echo "ffmpeg -i $f -vf fps=23 $IMAGE_FOLDER$fff-%d.png"
  ffmpeg -i $f -vf fps=23 $IMAGE_FOLDER$fff-%d.png
done