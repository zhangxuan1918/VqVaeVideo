# convert mkv to mp4
ffmpeg -i <VIDEO>.mkv -codec copy <VIDEO>.mp4

# clean up video, cut the beginning and the end of the video
ffmpeg -ss 00:03:49 -to 00:53:35 -i e01.mp4 -c copy e01_clean.mp4

# change resolution
ffmpeg -i e01_clean.mp4 -vf scale=256:144 e01_256_144.mp4