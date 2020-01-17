import os
import glob
import torch
from torchaudio import load
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset
  
class AudioDataset(Dataset):
    """Audio samples files dataset"""
    
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
    """Original MIDI files dataset"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            - root_dir (string): Directory with all the audio files.
            - transform (callable, optional): Optional transform to be applied on a sample
        """
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
            #print(midi.get_end_time()) # TO BE DELETED
            midi = self.transform(midi)
        
        return midi, label

class Snippets(Dataset):
    """Tensor snippets dataset"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = []
        
        for filename in glob.glob(os.path.join(root_dir, '*.pt')):
            self.labels.append(
                os.path.splitext(os.path.split(filename)[1])[0]
            )
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.labels[idx]
        filename = self.root_dir + '/' + label + '.pt'
        snippet = torch.load(filename)
        
        if self.transform:
            snippet = self.transform(snippet)
        
        return snippet, label


class PairSnippets(Dataset):
    """Pair of matching snippets dataset.
    Attributes:
        - midi_dataset (Dataset) : midi snippets dataset.
        - audio_dataset (Dataset) : audio snippets dataset.
        - labels (list) : list of common snippets labels.
        - reverse_dict (dict) : get pair index from label.
    """
    
    def __init__(self, midi_snippets_dataset, audio_snippets_dataset):
        self.midi_dataset = midi_snippets_dataset
        self.audio_dataset = audio_snippets_dataset
        
        midi_labels = midi_snippets_dataset.labels
        audio_labels = audio_snippets_dataset.labels
        
        midi_idxs  = list(range(len(midi_snippets_dataset)))
        audio_idxs = list(range(len(audio_snippets_dataset)))

        midi_dict  = dict(zip(midi_labels,  midi_idxs))
        audio_dict = dict(zip(audio_labels, audio_idxs))
        
        self.multimod_idxs = []
        self.labels = list(set(midi_labels).intersection(audio_labels))
        
        for label in self.labels:
            self.multimod_idxs.append([midi_dict[label], audio_dict[label]])
            
        pair_idxs = list(range(len(self.labels)))
        
        self.reverse_dict = dict(zip(self.labels, pair_idxs))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.labels[idx]
        midi_idx, audio_idx = self.multimod_idxs[idx]
        
        return  self.midi_dataset[midi_idx][0], \
                self.audio_dataset[audio_idx][0], \
                label
        
        