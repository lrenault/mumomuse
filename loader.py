import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets

class AudioLoader():
    '''Audio dataset loader'''
    def __init__(self):
        self.target_sr = 22050
        self.preproc = transforms.Compose([
                torchaudio.transforms.Resample(44100, self.target_sr),
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.target_sr,
                    n_fft=2048,
                    win_length=2048,
                    f_min=30.0,
                    f_max=6000.0,
                    n_mels=92),
                # crop the end
                lambda x: x[:,:,:42],
                transforms.Normalize([0], [1]),
        ])
        
    def getYESNOdata(self):
        '''Return data from the YESNO database'''
        data = torchaudio.datasets.YESNO(
            "db/",
            transform=self.preproc,
            download=True)
        return data
    
    def getYESNOLoader(self):
        '''Return loader for the YESNO database'''
        data = self.getYESNOdata()
        loader = DataLoader(data, batch_size=1)
        return loader
    
    #def split(self, dataset, max_time_length=42):
        
    
    
class MIDILoader():
    '''MIDI file loader'''
    def __init_(self):
        self.frame_rate = 21.54
        self.preproc_stack = transforms.Compose([
                lambda x: self.getPianoRoll(x)
        ])
        self.preproc_unstack = transforms.Compose([
                lambda x: self.getPianoRoll(x, stack=False)
        ])
    
    def getPianoRoll(self, midi, stack=True):
        """
        Args:
            - midi (pretty_midi) : midi data
            - stack (bool) : stack all insturment into 1 roll or not
        """
        nb_instru = 0
        for instrument in midi.instruments:
            if not instrument.is_drum:
                instruRoll = instrument.get_piano_roll(fs=self.frame_rate)
                if nb_instru == 0:
                    if stack:
                        data = instruRoll
                    else:
                        data = [instruRoll]
                else :
                    if stack:
                        data += instruRoll
                    else:
                        data.append(instruRoll)
                nb_instru += 1
        data = torch.Tensor(data)
        return data
                
    
    def loader(self, root_dir, batch_size=1, stackInstruments=True):
        """
        Args:
            - dataset (torch.utils.data.Dataset): raw midi dataset.
            - batch_size (int) : loading batch size.
            - stackInstruments (bool): stack all instruments in 1 pianoroll or not
        """
        if stackInstruments:
            dataset = datasets.MIDIDataset(
                    root_dir,
                    transform=self.preproc_stack)
        else:
            dataset = datasets.MIDIDataset(
                    root_dir,
                    transform=self.preproc_unstack)
            
        loader = DataLoader(dataset, batch_size=batch_size)
        return loader
    
    def split_and_export(self, root_dir, max_time_bin=42):
        """
        Args:
            - root_dir (string) : folder containing raw midi files.
            - max_time_bin (int) : maximum time bin for preprocessed piano roll tensors.
        """
        return None
