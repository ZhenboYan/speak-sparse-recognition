#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: ./record_audio.sh <duration> <file_suffix>"
  exit 1
fi

echo "Recording..."
ffmpeg -f avfoundation -i ":$3" -t "$1" "./train/$2.wav"
echo "Finished recording"
