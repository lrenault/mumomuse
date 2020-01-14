import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from numpy import pad

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
                transforms.Normalize([0], [1]),
                ])
        
    def get_YESNOdata(self):
        '''Return data from the YESNO database'''
        data = torchaudio.datasets.YESNO(
            "db/",
            transform=self.preproc,
            download=True)
        return data
    
    def get_YESNOLoader(self):
        '''Return loader for the YESNO database'''
        data = self.get_YESNOdata()
        loader = DataLoader(data, batch_size=1)
        return loader
    
    #def split(self, dataset, max_time_length=42):
        
    
    
class MIDILoader():
    '''MIDI file loader'''
    def __init__(self):
        self.frame_rate = 21.54
        self.preproc_stack = transforms.Compose([
                lambda x: self.get_PianoRoll(x)
                ])
        self.preproc_unstack = transforms.Compose([
                lambda x: self.get_PianoRoll(x, stack=False)
                ])
    
    def get_PianoRoll(self, midi, stack=True):
        """
        Args:
            - midi (pretty_midi) : midi data.
            - stack (bool) : if True, stack all insturment into 1 piano roll.
        """
        nb_instru = 0
        length = 0
        for instrument in midi.instruments:
            if not instrument.is_drum:
                instruRoll = instrument.get_piano_roll(fs=self.frame_rate)
                instruLen = instruRoll.shape[1]
                if nb_instru == 0:
                    if stack:
                        data = instruRoll
                        length = instruLen
                    else:
                        data = [instruRoll]
                else :
                    if stack:
                        if instruLen > length: # instrument score longer than whole track length
                            data = pad(
                                    data,
                                    ((0, 0), (0, instruLen - length))
                                    )
                            length = instruLen
                            
                        if instruLen < length: # instrument score shorter than whole track length
                            instruRoll = pad(
                                    instruRoll,
                                    ((0, 0), (0, length - instruLen))
                                    )
                        data += instruRoll
                    else:
                        data.append(instruRoll)
                nb_instru += 1
        data = torch.Tensor(data)
        return data
                
    
    def loader(self, root_dir, batch_size=1, stackInstruments=True):
        """
        Args:
            - root_dir (torch.utils.data.Dataset): raw midi dataset path.
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
    
    def split_and_export(
            self,
            midi,
            music_name,
            max_time_bin=42,
            export_dir='db/splitMIDI/'
            ):
        """
        Args:
            - midi (tensor) : midi tensor to split.
            - music_name (string) : 
            - max_time_bin (int) : maximum time bin for exported piano roll tensors.
            - export_dir (string) : export folder path.
        """
        total_length_bin = midi.size()[2]
        nb_snippets = total_length_bin // max_time_bin
        
        for i in range (nb_snippets):
            snippet = midi[:, :, i * max_time_bin : (i + 1) * max_time_bin]
            torch.save(snippet, export_dir + music_name + '_' + str(i) + '.pt')   
        return None
    
    def split_and_export_dataset(
            self,
            root_dir,
            stackInstruments=True,
            max_time_bin=42,
            export_dir='db/splitMIDI/'
            ):
        """
        Import a MIDI dataset, transform, split and exports its files into inputtable tensors.
        Args:
            - root_dir (string) : folder containing raw midi files.
            - stackInstruments (bool): if True, stack all instruments into 1 pianoroll.
            - max_time_bin (int) : maximum time bin for exported piano roll tensors.
            - export_dir (string) : export folder path.
        """
        midi_loader = self.loader(root_dir, batch_size=1, stackInstruments=stackInstruments)
        for midi, music_name in midi_loader:
            self.split_and_export(
                    midi,
                    music_name[0],
                    max_time_bin=max_time_bin,
                    export_dir=export_dir
                    )
            print(music_name, 'splitted and exported.')
        return None
        
    def midi_snippets_loader(self, batch_size=1, shuffle=False, root_dir='db/splitMIDI'):
        """ MIDI snippets tensors loader """
        dataset = datasets.Snippets(root_dir)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
