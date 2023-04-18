#!/bin/bash
echo "Recording..."
ffmpeg -f avfoundation -i ":0" -t $1 $2.wav
echo "Finished recording"

