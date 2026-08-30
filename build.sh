#!/usr/bin/env bash
set -e

pip install -r requirements.txt

mkdir -p data

MODEL="data/shape_predictor_68_face_landmarks.dat"
if [ ! -f "$MODEL" ]; then
  echo "Downloading dlib face landmark model..."
  curl -L "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" \
    -o shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
  mv shape_predictor_68_face_landmarks.dat "$MODEL"
  echo "Model downloaded."
fi
