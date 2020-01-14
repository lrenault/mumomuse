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

audiopath = 'db/nottingham-dataset-master/AUDIO'
MIDIpath = 'db/nottingham-dataset-master/MIDI'

audio_loader = loader.AudioLoader()
midi_loader = loader.MIDILoader()

raw_audio_loader = audio_loader.loader(audiopath)
raw_midi_loader  = midi_loader.loader(MIDIpath)

music_names = raw_midi_loader.dataset.music_names
idxs = list(range(len(raw_midi_loader.dataset)))

correspondance_dict = dict(zip(music_names, idxs))

k = 0
for batch_audio, batch_music_name in raw_audio_loader:
    midi, label = raw_midi_loader.dataset[correspondance_dict[batch_music_name[0]]]

    audio_length = batch_audio.squeeze(0).size()[2]
    midi_length = midi.size()[2]
    
    print(audio_length//42 - midi_length//42, audio_length, midi_length)
    
    if k == 100:
        break
    k += 1

#%%
midi_loader.split_and_export_dataset(MIDIpath)
midi_dataset = datasets.Snippets('db/splitMIDI')

audio_loader.split_and_export_dataset(audiopath)
audio_dataset = datasets.Snippets('db/splitAUDIO')
#%%
MIDIsnippet_loader = midi_loader.midi_snippets_loader(batch_size=1, shuffle=True)
for truc, name in MIDIsnippet_loader:
    print(truc.size())
    break
print(MIDIsnippet_loader.dataset[3][0].size())

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
filename = 'db/nottingham-dataset-master/AUDIO/ashover3.wav'
# load audio
waveform, origin_sr = torchaudio.load(filename)
waveform = waveform.mean(0).unsqueeze(0)
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
