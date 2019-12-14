#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:44:40 2019

@author: lrenault
"""
#%% imports
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

target_sr = 22050
filename = 'db/test.wav'

preproc = transforms.Compose([
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

yesno_data = torchaudio.datasets.YESNO("db/",
                                       transform=preproc,
                                       download=True)
data_loader = torch.utils.data.DataLoader(yesno_data, batch_size=1)
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

#%% network definitions
class audio_autoencoder(nn.Module):
    def __init__(self):
        
        super(audio_autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Conv2d(1,  24, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(24),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(48),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32)
                )
        
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 96, kernel_size=1, stride=1, padding=0),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(96, 48, kernel_size=3, stride=1, padding=1),
                # (1, 48, 23, 11)
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1,
                                                          output_padding=1),
                nn.ConvTranspose2d(48, 24, kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(24, 24, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(24, 1,  kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True)
                )
        
    def forward(self, x):
        x = self.encoder(x)
        
        # reshaping
        I, J = x.size()[2], x.size()[3]
        x = x.view(-1, 32 * I *J)
        
        # FC layers
        x = nn.Linear(32 * I * J, 128)(x)
        x = nn.Linear(128, 32)(x)
        
        # Average Pooling
        L = nn.AvgPool1d(kernel_size=1)(x.unsqueeze(1))
        #print("Latent dimension =", L.size())
        
        # FC Layers
        y = nn.Linear(32,  128)(L)
        y = nn.Linear(128, 32 * I * J)(y)
        
        # reshaping for decoder
        y = y.view(1, 32, I, J)
        
        # decoding
        x_hat = self.decoder(y)
        x_hat = x_hat[:, :, :92, :42]

        return x_hat

# Optimization definition
num_epochs = 20
batch_size = 100
learning_rate = 2e-3
    
    
model = audio_autoencoder()
#print(model)

criterion = nn.functional.mse_loss
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)

#%% Optimization run
writer = SummaryWriter()

for epoch in range(num_epochs):
    for data, label in data_loader:
        spec = data
        # ===== forward  =====
        output = model(spec)
        #print("Sizes:", spec.size(), output.size())
        loss   = criterion(output, spec)

        # ===== backward =====
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # ===== log =====
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1,
                                              num_epochs,
                                              loss.data.item()))
    img = output.squeeze(1)
    writer.add_image('epoch'+str(epoch), img)
    
    plt.figure(figsize=(10,5))
    plt.imshow(output[0,0,:,:].detach().numpy(), origin='lower')
    plt.show()

writer.close()    
#%%
torch.save(model.state_dict(), './models/audio_AE.pth')