import os
import glob
import torch
from torchaudio import load
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset
  
class AudioDataset(Dataset):
    """Raw audio samples files dataset.
    Attributes:
        - root_dir (path): path to the folder containing the Audio dataset.
        - transform (callable): transform to apply on data.
        - music_names (list): list of music names.
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.music_names = []
        
        for filename in glob.glob(os.path.join(root_dir, '*.wav')):
            self.music_names.append(
                os.path.splitext(os.path.split(filename)[1])[0]
            )
    
    def __len__(self):
        return len(self.music_names)
    
    def __getitem__(self, idx):
        """
        Return audio data (tensor) and label (string).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.music_names[idx]
        filename = self.root_dir + '/' + label + '.wav'
        audio = load(filename)
        
        if self.transform:
            audio = self.transform(audio)
        
        return audio, label


class MIDIDataset(Dataset):
    """Raw MIDI files dataset.
    Attributes:
        - root_dir (string): Directory with all the audio files.
        - transform (callable): transform to be apply on a data.
        - music_names (list): list of music names.
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.music_names = []
        
        for filename in glob.glob(os.path.join(root_dir, '*.mid')):
            self.music_names.append(
                os.path.splitext(os.path.split(filename)[1])[0]
            )
        
    
    def __len__(self):
        return len(self.music_names)
    
    def __getitem__(self, idx):
        """
        Return midi object (pretty_midi) and label (string).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.music_names[idx]
        filename = self.root_dir + '/' + label + '.mid'
        midi = PrettyMIDI(filename)
        
        if self.transform:
            midi = self.transform(midi)
        
        return midi, label