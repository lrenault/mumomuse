import torch
import torchaudio
import torchaudio.transforms
from torchvision import transforms


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
        transforms.ToPILImage(),
         k = lambda x: x.size()[2] // 42,
        for i in range(k):
            lambda x : functional.crop(x,1,i*42,92,42),
        lambda x: x[:,:,:42],
        transforms.Normalize([0], [1]),
        ])
        transforms.ToTensor(),
         
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
    
    
class MIDILoader():
    '''MIDI file loader'''
    def __init_(self):
        return None
