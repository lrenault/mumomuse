import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import datasets
import loader

MIDI_MODE = True
small_test = True

if MIDI_MODE:
    dataset = datasets.Snippets('db/splitMIDI')
else:
    audio_loader = loader.AudioLoader()
    dataset = audio_loader.getYESNOdata()

# train-test split
set_size   = len(dataset)
train_size = int(set_size * 0.8)
train_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_size, set_size - train_size])

if small_test:
    train_set, train_leftlovers = torch.utils.data.random_split(
            train_set,
            [320, len(train_set) - 320])
    test_set, test_leftlovers = torch.utils.data.random_split(
            test_set,
            [64, len(test_set) - 64])

# loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=1)

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
        x = x.view(-1, 32*5*2)          # reshaping
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
        y = y.view(1, 32, 5, 2)             # reshaping
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
        x = x.view(-1, 32*8*2)          # reshaping
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
        y = y.view(1, 32, 8, 2)             # reshaping
        x_hat = self.decoder(y)
        #print("Reconstructed dimension:", x_hat.size())
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
        
#%% train and test definition
def train(model, train_loader, optimizer, epoch):
    model.train()
    for data, label in train_loader:
        # forward
        output = model(data)
        loss   = criterion(output, data)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print('epoch [{}], loss:{:.4f}'.format(epoch+1, loss.data.item()))

def test(model, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, label in test_loader:
            output = model(data)
            test_loss += criterion(output, data).data.item()
            
    test_loss /= len(test_loader.dataset)
    writer.add_scalar('Test loss', test_loss, epoch)
    
    img = output.squeeze(0)
    writer.add_image('epoch'+str(epoch), img)
    plt.imshow(img.squeeze(0))
    plt.show()

#%%
# Optimization definition
num_epochs = 15
batch_size = 100
learning_rate = 2e-3

if MIDI_MODE:
    model = midi_AE()
else:
    model = audio_AE()

criterion = nn.functional.mse_loss
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)
#%% Optimization run
writer = SummaryWriter()

for epoch in range(num_epochs):
    train(model, train_loader, optimizer, epoch)
    print('epoch [{}], end of training.'.format(epoch+1))
    test(model, test_loader, writer, epoch)
    print('epoch [{}], end of testing'.format(epoch+1))

writer.close()    
#%%
if MIDI_MODE:
    torch.save(model.state_dict(), './models/midi_AE.pth')
else:
    torch.save(model.state_dict(), './models/audio_AE3.pth')