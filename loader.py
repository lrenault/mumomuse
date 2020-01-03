import torch
import torchaudio
import torchaudio.transforms
from torchvision import transforms
from torch.utils.data import DataLoader

import datasets

class AudioLoader():
    '''Audio dataset loader'''
    target_sr = 22050
    
    def __init__(self):
        self.preproc = transforms.Compose([
        lambda x: torchaudio.transforms.Resample(44100, self.target_sr)(x),
        lambda x: torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_sr,
                n_fft=len(x.t().numpy()),
                win_length=512, # paper said : 2048
                f_min=30.0,
                f_max=6000.0,
                n_mels=92)(x),
        #transforms.ToTensor(),
        # crop the end
        transforms.Normalize([0], [1]),
        ])
        
    def loader(self, root_dir, batch_size=1):
        """
        Args:
            - root_dir (torch.utils.data.Dataset): audio dataset path.
            - batch_size (int) : loading batch size.
        """
        dataset = datasets.audioDataset(root_dir, transform=self.preproc)      
        loader = DataLoader(dataset, batch_size=batch_size)
        return loader
         
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
        loader = torch.utils.data.DataLoader(data, batch_size=1)
        return loader
    
    def split(self, spectro, name, max_time = 42, export_dir = 'db/splitAudio/'):
        length_sp = spectro.shape()[2]
        n_snips = length_sp // max_time
        for i in range(n_snips):
            CropSpectro = transforms.functional.crop(spectro,1,i*42,92,42)
            snip = transforms.ToTensor(CropSpectro)
            torch.save(snip, export_dir + name + '_' + str(i) + '.pt')    
        return None
    
    def splitData(self,root,max_time = 42,export_dir = 'db/splitAudio'):
        audio_loader = self.loader(root)
        for spectro, name in audio_loader:
            self.split(
                    spectro,
                    name[0],
                    max_time=max_time,
                    export_dir=export_dir
                    )
            print(name, 'splitted and exported.')
        return None
        
    def audio_snippets_loader(self, batch_size=1, root_dir='db/splitAudio'):
        """ Audio snippets tensors loader """
        dataset = datasets.Snippets(root_dir)
        loader  = DataLoader(dataset, batch_size=batch_size)
        return loader
    
class MIDILoader():
    '''MIDI file loader'''
    def __init_(self):
        return None
