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

import loader

target_sr = 22050
filename = 'db/test.wav'

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
                                                n_fft=len(waveform.t().numpy()),
                                                win_length=2048,
                                                f_min=30.0,
                                                f_max=6000.0,
                                                n_mels=92)(waveform)

data_m = specgram[:,:,:42]
print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure(figsize=(15,5))
plt.imshow(data_m.log2()[0,:,:].detach().numpy(), origin='lower')
