import torch
import torchaudio
import pretty_midi
from torch.utils.data import Dataset

class MIDIDataset(Dataset):
    """Original MIDI file dataset"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            - root_dir (string): Directory with all the audio files.
            - transform (callable, optional): Optional transform to be applied on a sample
        """
        self.music_name = None #TBC
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.music_name)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get path for indicated file
        filename = 'db/nottingham-dataset-master/MIDI/ashover3.mid'
        midi = pretty_midi.PrettyMIDI(filename)
        
        if self.transform:
            midi = self.transform(midi)
        
        return midi
    
class AudioDataset(Dataset):
    """Audio samples file dataset"""
    
    def __init__(self, root_dir, transform=None):
        self.music_name = None #TBC
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.music_name)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get path for indicated file
        filename = 'db/nottingham-dataset-master/Audio/ashover3.wav'
        audio = torchaudio.load(filename)
        
        if self.transform:
            audio = self.transform(audio)
        
        return audio
        