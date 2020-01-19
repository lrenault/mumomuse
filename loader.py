import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from numpy import pad

import numpy as np # TO BE DELETED

import datasets
import utils

class AudioLoader():
    '''Audio dataset loader class
        Instance variables :
            - target_sr : Sample rate at which the audio should be resampled
        Methods :
            - loader : Loads the dataset comprised of the full length .wav files
            - split : Splits a PIL image into smaller tensors
            - splitData : Creates a set of tensors extracted from a set of PIL images
            - audio_snippets_loader : Loads a set of spectrogram snippets 
    '''
    def __init__(self):
        self.target_sr = 22050
        self.preproc = transforms.Compose([
                lambda x: torchaudio.transforms.Resample(x[1],
                                                         self.target_sr)(x[0]),
                transforms.Lambda(lambda x: x.mean(0).unsqueeze(0)), # convert to mono
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.target_sr,
                    n_fft=2048,
                    win_length=2048,
                    f_min=30.0,
                    f_max=6000.0,
                    n_mels=92),
                transforms.Normalize([0], [1]),
                ])
        
    def loader(self, root_dir, batch_size=1):
        """
        Args:
            - root_dir (torch.utils.data.Dataset): audio dataset path.
            - batch_size (int) : loading batch size.
        """
        dataset = datasets.AudioDataset(root_dir, transform=self.preproc)      
        loader = DataLoader(dataset, batch_size=batch_size)
        return loader
    
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

    def split_and_export(self, spectro, name, max_time=42, 
                         export_dir='db/splitAUDIO'):
        """
        Splits a spectrogram  into tensors corresponding to audio snippets from the input. 
        Args:
            - spectro (PIL image) : spectrogram image to split.
            - name (string) : name of the piece correspondong the input spectrogram PIL image.
            - max_time_bin (int) : maximum time bin for the exported spectrogram tensors.
            - export_dir (string) : export folder path.
        """
        length_sp = spectro.size()[2]
        n_snips = length_sp // max_time
        
        for i in range(n_snips):
            snip = spectro[:, :, i*max_time : (i + 1) * max_time]
            torch.save(snip, export_dir + '/' + name + '_' + str(i) + '.pt')    
        return None
    
    def split_and_export_dataset(self, root, max_time=42,
                                 export_dir='db/splitAUDIO'):
        """
        Import an audio dataset, transform, split and exports its files into inputtable tensors.
        Args:
            - root (string) : folder containing raw wav files.
            - max_time (int) : maximum time bin for exported spectrograms.
            - export_dir (string) : export folder path.
        """
        audio_loader = self.loader(root, batch_size=1)
        print("Splitting Audio dataset...")
        for spectro, name in audio_loader:
            self.split_and_export(spectro.squeeze(0), name[0],
                                  max_time=max_time,
                                  export_dir=export_dir)
        print('Audio dataset splitted and exported.')
        return None
        
    def audio_snippets_loader(self, batch_size=1, root_dir='db/splitAUDIO/'):
        """ 
        Audio snippets tensors loader
        Args :
            - batch_size (int) : number of snippets that will be loaded
            - root_dir (string) : dataset path
        """
        dataset = datasets.Snippets(root_dir)
        loader  = DataLoader(dataset, batch_size=batch_size)
        #self.addNoise(loader)
        return loader
        
    
    def addNoise(self, loader, augmented_proportion=0.1, noise_level=0.1):
        """
        Adds noise to a certain proportion of the dataset.
        Args :
            - augmented_proportion (float) = Proportion of snippets to add noise to.
            - noise_level (float) = Level of noise to be added.
            - loader (torch.utils.data.DataLoader) = Loader of the dataset to add noise to.
        """
        for snip,name in loader:
            
            augment = np.random.random() # use torch.random instead
            
            if(augment < augmented_proportion):  
                npSnip = snip.numpy()
                noise = np.random.random(size = npSnip.shape())
                noiseSnipNP = npSnip + noise
                snip = torch.from_numpy(noiseSnipNP)              
  
        return None
    
    
    
class MIDILoader():
    '''MIDI files loader class
        Attributes :
            - preproc_stack (transforms) : pre-processing transformations to stack instruments into 1 channel.
            - preproc_unstack (transforms) : pre-processing transformations while keeping instruments into different channels.
    '''
    def __init__(self):
        self.preproc_stack = transforms.Compose([
                lambda x: self.get_PianoRoll(x),
                ])
        self.preproc_unstack = transforms.Compose([
                lambda x: self.get_PianoRoll(x, stack=False)
                ])
    
    def get_PianoRoll(self, midi, stack=True):
        """
        Args:
            - midi (pretty_midi) : midi data.
            - stack (bool) : if True, stack all insturment into 1 piano roll.
        Out:
            data (C, 128, L_audio) : piano roll of the midi file.
        """
        nb_instru = 0
        length = 0
        fs = utils.sampling_period_from_length(midi.get_end_time())
        for instrument in midi.instruments:
            if not instrument.is_drum:
                instruRoll = instrument.get_piano_roll(fs=fs)
                instruLen = instruRoll.shape[1]
                if nb_instru == 0:              # init
                    if stack:
                        data = instruRoll
                        length = instruLen
                    else:
                        data = [instruRoll]
                else :
                    if stack:
                        if instruLen > length:  # instrument score longer than whole track length
                            data = pad(
                                    data,
                                    ((0, 0), (0, instruLen - length))
                                    )
                            length = instruLen
                            
                        if instruLen < length:  # instrument score shorter than whole track length
                            instruRoll = pad(
                                    instruRoll,
                                    ((0, 0), (0, length - instruLen))
                                    )
                        data += instruRoll
                    else:
                        data.append(instruRoll)
                nb_instru += 1
        # tensor conversion
        data = torch.Tensor(data)
        if stack:
            data = data.unsqueeze(0)            # add channel dimension
        return data
                
    
    def loader(self, root_dir, batch_size=1, stackInstruments=True):
        """
        Args:
            - root_dir (torch.utils.data.Dataset): raw midi dataset path.
            - batch_size (int) : loading batch size.
            - stackInstruments (bool): stack all instruments in 1 pianoroll or not
        """
        if stackInstruments:
            dataset = datasets.MIDIDataset(root_dir,
                                           transform=self.preproc_stack)
        else:
            dataset = datasets.MIDIDataset(root_dir,
                                           transform=self.preproc_unstack)
            
        loader = DataLoader(dataset, batch_size=batch_size)
        return loader
    
    def split_and_export(self, midi, music_name, max_time_bin=42,
                         export_dir='db/splitMIDI'):
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
            torch.save(snippet, export_dir + '/' + music_name + '_' + str(i) + '.pt')   
        return None
    
    def split_and_export_dataset(self, root_dir, export_dir='db/splitMIDI',
                                 stackInstruments=True, max_time_bin=42):
        """
        Import a MIDI dataset, transform, split and exports its files into inputtable tensors.
        Args:
            - root_dir (string) : folder containing raw midi files.
            - export_dir (string) : export folder path.
            - stackInstruments (bool): if True, stack all instruments into 1 pianoroll.
            - max_time_bin (int) : maximum time bin for exported piano roll tensors.
        """
        midi_loader = self.loader(root_dir, batch_size=1, stackInstruments=stackInstruments)
        print("Splitting MIDI dataset...")
        
        for midi, music_name in midi_loader:
            self.split_and_export(midi.squeeze(0), music_name[0], 
                                  max_time_bin=max_time_bin,
                                  export_dir=export_dir)        
        print("MIDI dataset splitted and exported.")
        return None
        
    def midi_snippets_loader(self, batch_size=1, shuffle=False, root_dir='db/splitMIDI'):
        """ MIDI snippets tensors loader """
        dataset = datasets.Snippets(root_dir)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader



def get_dictionnary(midi_dataset, audio_dataset):
    """Construct the dictionnary retrieving an audio snippets from its label.
    Args:
        - audio_dataset (Dataset): audio snippet dataset.
        - midi_dataset (Dataset): corresponding midi_dataset.
    Out:
        - correspondance_dict (dict): audio-midi matching dictionnary.
    """
    music_names = audio_dataset.labels
    idxs = list(range(len(audio_dataset)))

    correspondance_dict = dict(zip(music_names, idxs))
    '''
    midi_loader = DataLoader(midi_dataset, batch_size=10)
    
    for batch_midi_snip, batch_labels in midi_loader:
        for label in batch_labels:
            try:
                prout = correspondance_dict[label]
            except KeyError:
                print("Non-correspondance found :", label)
                previous_label = utils.previous_label(label)
                idxs.append(correspondance_dict[previous_label])
                music_names.append(label)
    correspondance_dict = dict(zip(music_names, idxs))
    '''
    print("Finished correspondance dictionnary construction.")
    return correspondance_dict