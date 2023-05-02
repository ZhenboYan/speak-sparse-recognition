#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: ./record_audio.sh <duration> <file_suffix>"
  exit 1
fi

cd ./test
mv *.wav ./not_tested
cd ..

echo "Recording..."
ffmpeg -f avfoundation -i ":$3" -t "$1" "./test/$2.wav"
echo "Finished recording"

python3.8 predict.py