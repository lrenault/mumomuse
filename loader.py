import torchaudio
import torchaudio.transforms
from torch.utils.data import DataLoader
from torchvision import transforms


class AudioLoader():
    '''Audio dataset loader'''
    target_sr = 22050
    
    def __init__(self):
        self.preproc = transforms.Compose([
                torchaudio.transforms.Resample(44100, self.target_sr),
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.target_sr,
                    n_fft=2048,
                    win_length=2048, # paper said : 2048
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
    
    
class MIDILoader():
    '''MIDI file loader'''
    def __init_(self):
        return None
