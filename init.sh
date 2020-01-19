#!/bin/bash

mkdir temp
mkdir db
cd db
mkdir splitAUDIO
mkdir splitMIDI
cd ..

pip install torch torchvision
pip install tensorboard
pip install pretty_midi
# torchaudio install
apt-get install sox libsox-dev libsox-fmt-all
pip install git+git://github.com/pytorch/audio
