#!/bin/bash

mkdir temp
mkdir db
cd db
mkdir splitAUDIO
mkdir splitMIDI
cd ..

python3 -m pip install torch torchvision
python3 -m pip install pretty_midi
python3 -m pip install tensorboard
# torchaudio install
apt-get install sox libsox-dev libsox-fmt-all
python 3 -m pip install git+git://github.com/pytorch/audio
