#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:44:40 2019

@author: lrenault
"""
#%% imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


filename = "db/test.wav"
waveform, sr_hz = torchaudio.load(filename)

# convert to mono
#if waveform.size()[0] > 1:
#    waveform = waveform[0,:]

plt.plot(waveform.t().numpy())
plt.show()

specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr_hz,
                                                f_min=30.0,
                                                f_max=6000.0)(waveform)
print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure(figsize=(30,10))
plt.imshow(specgram.log2()[0,:,:].detach().numpy())

#%% network definitions
class audio_autoencoder(nn.Module):
    def __init__(self):
        super(audio_autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Conv2d(92*42, 24, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(100),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(100),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(100),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(100),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(100),
                nn.Linear(32, 32),
                
                nn.AvgPool2d(kernel_size=1)
                )
        
        self.decoder = nn.Sequential(
                nn.Linear(32, 32),
                nn.ConvTranspose2d(32, 96, kernel_size=1, stride=3, padding=0),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(48, 48, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(24, 24, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(24, 92*42, kernel_size=3, stride=1, padding=3)
                )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = audio_autoencoder()
print(model)