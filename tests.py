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

x = []
y = []
y2= []

k = 0
for batch_audio, batch_music_name in raw_audio_loader:
    midi, label = raw_midi_loader.dataset[correspondance_dict[batch_music_name[0]]]

    audio_length = batch_audio.squeeze(0).size()[2]
    midi_length = midi.size()[2]
    
    print(midi_length//42 - audio_length//42, audio_length, midi_length)
    
    x.append(audio_length)
    y.append(midi_length - audio_length)
    y2.append(midi_length//42 - audio_length//42)
    
    if k == 100:
        break
    k += 1
#%%
error = np.argmax(x)
del x[error]
del y[error]
del y2[error]
#%%
plt.figure(figsize=(10, 10))
plt.plot(x, y, 'o')
plt.plot(x, np.zeros(len(x)), color='r')
plt.xlabel('Audio length [time bin]')
plt.ylabel('(Pianoroll length) - (Audio length) [time bin]')
plt.title('Alignment Error (with adaptative sampling period)')
plt.savefig('alignment_error_adaptative.png')
plt.show()
#%%
plt.figure(figsize=(10,10))
plt.plot(x, np.zeros(len(x)), color='r')
plt.plot(x, y2, 'o')
plt.show()
#%%
plt.figure(figsize=(10, 10))
t_v = np.arange(770)
y_v = [utils.sampling_period_from_length(t) for t in t_v]
plt.plot(t_v, y_v)
plt.show()
#%%
plt.figure(figsize=(10, 10))
plt.plot(x21, y21, 'o', label='Ts=21.43 ms')
plt.plot(x22, y22, 'o', label='Ts=22.43 ms')
plt.plot(x23, y23, 'o', label='Ts=23.43 ms')
plt.plot(x21, np.zeros(len(x21)), color='r')
plt.title('Alignment Error (with fixed sampling period Ts)')
plt.xlabel('Audio length')
plt.ylabel('Pianoroll length - Audio length')
plt.legend()
plt.savefig('alignement_error_fixed.png')
plt.show()

#%%
midi_loader.split_and_export_dataset(MIDIpath)
midi_dataset = datasets.Snippets('db/splitMIDI')
#%%
audio_loader.split_and_export_dataset(audiopath)
audio_dataset = datasets.Snippets('db/splitAUDIO')
#%%
MIDIsnippet_loader = midi_loader.midi_snippets_loader(batch_size=1, shuffle=True)
for truc, name in MIDIsnippet_loader:
    print(truc.size())
    break
print(MIDIsnippet_loader.dataset[3][0].size())
#%%
midi_test = 'db/nottingham-dataset-master/MIDI/ashover10.mid'
midi = pretty_midi.PrettyMIDI(midi_test)
print(midi.get_end_time())

#%%
filename = 'db/nottingham-dataset-master/AUDIO/ashover3.wav'
# load audio
waveform, origin_sr = torchaudio.load(filename)
waveform = waveform.mean(0).unsqueeze(0)
print(waveform.size())
# take only first channel and 22.05kHz sampling rate
waveform = torchaudio.transforms.Resample(origin_sr,
                                          22500)(waveform[0]) 

specgram = torchaudio.transforms.MelSpectrogram(sample_rate=22500,
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
