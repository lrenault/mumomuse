#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 15:34:28 2019

@author: lrenault
"""

import torch
import torchaudio
import torchaudio.transforms
from torchvision import transforms

target_sr = 22050

audio_preproc = transforms.Compose([
        lambda x: torchaudio.transforms.Resample(44100, target_sr)(x),
        lambda x: torchaudio.transforms.MelSpectrogram(sample_rate=target_sr,
                                                       n_fft=len(x.t().numpy()),
                                                       win_length=512, # paper said : 2048
                                                       f_min=30.0,
                                                       f_max=6000.0,
                                                       n_mels=92)(x),
        #transforms.ToTensor(),
        # crop the end
        lambda x: x[:,:,:42],
        transforms.Normalize([0], [1]),
        ])
        
        
def getYESNOdata():
    data = torchaudio.datasets.YESNO(
            "db/",
            transform=audio_preproc,
            download=True)
    return data