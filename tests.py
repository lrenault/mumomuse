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
import snippets
import preproc

#%% tests alignments
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

error = np.argmax(x)
del x[error]
del y[error]
del y2[error]

plt.figure(figsize=(10, 10))
plt.plot(x, y, 'o')
plt.plot(x, np.zeros(len(x)), color='r')
plt.xlabel('Audio length [time bin]')
plt.ylabel('(Pianoroll length) - (Audio length) [time bin]')
plt.title('Alignment Error (with adaptative sampling period)')
plt.savefig('alignment_error_adaptative.png')
plt.show()
#%% y=0 axis
plt.figure(figsize=(10,10))
plt.plot(x, np.zeros(len(x)), color='r')
plt.plot(x, y2, 'o')
plt.show()
#%% Test sampling from period
plt.figure(figsize=(10, 10))
t_v = np.arange(770)
y_v = [utils.sampling_period_from_length(t) for t in t_v]
plt.plot(t_v, y_v)
plt.show()

#%% tests preprocessings
midi_file = 'db/nottingham-dataset-master/MIDI/ashover3.mid'
audio_file= 'db/nottingham-dataset-master/AUDIO/ashover3.wav'

midi = pretty_midi.PrettyMIDI(midi_file)
audio = torchaudio.load(audio_file)

preprocessed_midi = preproc.midi_preproc(True)(midi)
preprocessed_audio= preproc.audio_preproc(22050)(audio)

fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True, figsize=(30,5))
ax0.imshow(preprocessed_midi[0].numpy(), origin='lower')
ax1.imshow(preprocessed_audio[0].detach().numpy(), origin='lower')
plt.savefig('db/bleh.jpg')
plt.show()

#%% tests with snippets
midi_dataset = snippets.Snippets('db/splitMIDI')
audio_dataset = snippets.Snippets('db/splitAUDIO')

pairs_dataset = snippets.PairSnippets(midi_dataset, audio_dataset)

pairs_loader = torch.utils.data.DataLoader(pairs_dataset, batch_size=1, shuffle=True)

k = 0
for batch_midi, batch_audio, batch_labels in pairs_loader:
    print(batch_midi.size(), batch_audio.size(), batch_labels)
    
    fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
    
    ax0.imshow(batch_midi[0,0].numpy(), origin='lower')
    ax1.imshow(batch_audio[0,0].detach().numpy(), origin='lower')
    plt.show()

    k += 1
    if k==4:
        break
