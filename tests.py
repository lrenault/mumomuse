#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 15:30:01 2019

@author: lrenault
"""
#%%
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms
from torchvision import transforms
import matplotlib.pyplot as plt
import pretty_midi

import loader
import datasets
import utils

MIDIpath = 'db/nottingham-dataset-master/MIDI'
midiLoader = loader.MIDILoader()
audioLoader= loader.AudioLoader()
rawMIDIloader = midiLoader.loader(MIDIpath, batch_size=1)
MIDIsnippet_loader = midiLoader.midi_snippets_loader(batch_size=1, shuffle=True)

AUDIOpath = 'db/nottingham-dataset-master/AUDIO'
loader = audioLoader.get_YESNOdata()

dataset = MIDIsnippet_loader.dataset
music_names = dataset.labels
idxs = list(range(len(dataset)))

correspondance_dict = dict(zip(music_names, idxs))
#%%
k = 0
for snippet, name in MIDIsnippet_loader:
    #print(snippet.size(), '\n', name)
    names = name
    break

#print(names)
excepts = [correspondance_dict[name] for name in names]

print(utils.random_except(4, [2], 10))
#%%
filename = 'db/nottingham-dataset-master/MIDI/reelsr-t64.mid'
target_sr = 22050

midi = pretty_midi.PrettyMIDI(filename)
nb_instru = 0
for instrument in midi.instruments:
    if not instrument.is_drum:
        roll = instrument.get_piano_roll(fs=21.54)
        plt.figure(figsize=(20,10))
        plt.imshow(roll)
        plt.show()
        print((roll.shape[1]))
        if nb_instru == 0:
            data = roll
        else :
            data += roll
        nb_instru += 1
        

data = torch.Tensor(data)
print(data.size())
#%%
audio_loader = loader.AudioLoader()
data_loader  = audio_loader.get_YESNOLoader()

#%%
filename = 'db/nottingham-dataset-master/AUDIO/ashover3.wav'
# load audio
waveform, origin_sr = torchaudio.load(filename)
#waveform = torch.mean(waveform, dim=0, keepdim=True)
waveform = waveform.mean(0).unsqueeze(0)
#waveform = waveform[0,:].view(1,-1)
print(waveform.size())
#%%
# take only first channel and 22.05kHz sampling rate
waveform = torchaudio.transforms.Resample(origin_sr,
                                          target_sr)(waveform[0,:].view(1,-1)) 

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
