import torch.nn as nn

#%% Audio networks definition
class audio_encoder(nn.Module):
    def __init__(self):
        super(audio_encoder, self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Conv2d(1,  24, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(24),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(48),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32)
                )
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 32*5*2)
        # FC layers
        x = nn.Linear(32*5*2, 128)(x)
        x = nn.Linear(128, 32)(x)
        L = nn.AvgPool1d(kernel_size=1)(x.unsqueeze(1))
        #print("Latent dimension =", L.size())
        return L

class audio_decoder(nn.Module):
    def __init__(self):        
        super(audio_decoder, self).__init__()
        
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 96, kernel_size=1, stride=1, padding=0),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(96, 48, kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1,
                                                          output_padding=1),
                nn.ConvTranspose2d(48, 24, kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(24, 24, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(24, 1,  kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True)
                )
        
    def forward(self, L):
        # FC Layers
        y = nn.Linear(32,  128)(L)
        y = nn.Linear(128, 32*5*2)(y)
        
        y = y.view(1, 32, 5, 2)
        x_hat = self.decoder(y)
        
        x_hat = x_hat[:, :, :92, :42]       #crop
        return x_hat

class audio_AE(nn.Module):
    def __init__(self):
        super(audio_AE, self).__init__()
        
        self.AE = nn.Sequential(
                audio_encoder(),
                audio_decoder(),
                )
    def forward(self, x):
        x_hat = self.AE(x)
        return x_hat

#%% MIDI networks definitnion
class midi_encoder(nn.Module):
    def __init__(self):
        super(midi_encoder, self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Conv2d(1,  24, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(24),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(48),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ELU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32)
                )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 32*8*2)
        # FC layers
        x = nn.Linear(32*8*2, 256)(x)
        x = nn.Linear(256, 128)(x)
        x = nn.Linear(128, 32)(x)
        L = nn.AvgPool1d(kernel_size=1)(x.unsqueeze(1))
        #print("Latent dimension =", L.size())
        return L

class midi_decoder(nn.Module):
    def __init__(self):
        super(midi_decoder, self).__init__()
        
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 96, kernel_size=1, stride=1, padding=0),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.ConvTranspose2d(96, 48, kernel_size=3, stride=1, padding=1),

                nn.ELU(inplace=True),
                nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(48, 24, kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(24, 24, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.ConvTranspose2d(24, 1,  kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True)
                )
    
    def forward(self, L):
        # FC Layers
        y = nn.Linear(32,  128)(L)
        y = nn.Linear(128, 256)(y)
        y = nn.Linear(256, 32*8*2)(y)
        
        y = y.view(1, 32, 8, 2)
        x_hat = self.decoder(y)
        
        x_hat = x_hat[:, :, :128, :]       #crop
        return x_hat
    
class midi_AE(nn.Module):
    def __init__(self):
        super(midi_AE, self).__init__()
        
        self.AE = nn.Sequential(
                midi_encoder(),
                midi_decoder(),
                )
    def forward(self, x):
        x_hat = self.AE(x)
        return x_hat

#%% multimodal network
class multimodal(nn.Module):
    def __init__(self):
        super(multimodal, self).__init__()
        
        self.f = midi_encoder()
        self.g = audio_encoder()
    
    def forward(self, midi, audio):
        x = self.f(midi)
        y = self.g(audio)
        return x, y