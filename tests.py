#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 15:30:01 2019

@author: lrenault
"""
#%%
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pretty_midi
import glob
import os

import loader
import datasets

path = 'db/nottingham-dataset-master/MIDI'
MIDIset = datasets.MIDIDataset(path)
#%%
for filename in glob.glob(os.path.join(path, '*.mid')):
    print(os.path.splitext(os.path.split(filename)[1])[0])
    #midi = pretty_midi.PrettyMIDI(filename)
    #print(midi)
#%%
filename = 'db/nottingham-dataset-master/MIDI/ashover3.mid'
target_sr = 22050

midi = pretty_midi.PrettyMIDI(filename)
data = []
for instrument in midi.instruments:
    if not instrument.is_drum:
        data.append(instrument.get_piano_roll(fs=21.54))

data = torch.Tensor(data)
print(data.size())
#%%
filename = 'db/nottingham-dataset-master/AUDIO/ashover3.wav'
#%%
audio_loader = loader.AudioLoader()
data_loader  = audio_loader.getYESNOLoader()
#%%
i=0
for truc, label in data_loader:
    print(truc.size())
    i+=1
    if i > 5:
        print(truc.size())
        test = nn.MaxPool2d(2)(truc)
        print(test.size())
        test2= nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=3, output_padding=0)(test)
        print(test2.size())
        break

#%%

# load audio
waveform, origin_sr = torchaudio.load(filename)
# take only first channel and 22.05kHz sampling rate
waveform = torchaudio.transforms.Resample(origin_sr,
                                          target_sr)(waveform[0,:].view(1,-1)) 

plt.plot(waveform.t().numpy())
plt.show()

specgram = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr,
                                                n_fft=2048,
                                                win_length=2048,
                                                f_min=30.0,
                                                f_max=6000.0,
                                                n_mels=92)(waveform)
print("specgram finished")
data_m = specgram[:,:,:42]
print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure(figsize=(15,5))
plt.imshow(data_m.log2()[0,:,:].detach().numpy(), origin='lower')
