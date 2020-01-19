import os
import glob
import torch
from torch.utils.data import Dataset

class Snippets(Dataset):
    """Tensor snippets dataset. Could be Audio snippet or MIDI snippet.
    Attributes:
        - root_dir (path): path to the folder containing the .pt files.
        - transform (torchvision.Transforms): transforms to apply on data.
        - labels (list): list of snippet labels.
    """
    
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
        """Retuns snippet object (tensor) and label (string)."""
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
        - midi_dataset (Snippets) : midi snippets dataset.
        - audio_dataset (Snippets) : audio snippets dataset.
        - labels (list) : list of common snippets labels.
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
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Return """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.labels[idx]
        midi_idx, audio_idx = self.multimod_idxs[idx]
        
        return  self.midi_dataset[midi_idx][0], \
                self.audio_dataset[audio_idx][0], \
                label