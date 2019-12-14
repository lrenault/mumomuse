import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import loader

audio_loader = loader.AudioLoader()
data_loader = audio_loader.getYESNOLoader()

#%% network definitions
class audio_autoencoder(nn.Module):
    def __init__(self):
        
        super(audio_autoencoder, self).__init__()
        
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
        
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 96, kernel_size=1, stride=1, padding=0),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(96, 48, kernel_size=3, stride=1, padding=1),
                # (1, 48, 23, 11)
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1,
                                                          output_padding=1),
                nn.ConvTranspose2d(48, 24, kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(24, 24, kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose2d(24, 1,  kernel_size=3, stride=1, padding=1),
                
                nn.ELU(inplace=True)
                )
        
    def forward_encoder(self, x):
        x = self.encoder(x)
        
        # reshaping
        x = x.view(-1, 32*5*2)
        
        # FC layers
        x = nn.Linear(32*5*2, 128)(x)
        x = nn.Linear(128, 32)(x)
        
        # Average Pooling
        L = nn.AvgPool1d(kernel_size=1)(x.unsqueeze(1))
        #print("Latent dimension =", L.size())
        return L
        
    def forward_decoder(self, L):
        # FC Layers
        y = nn.Linear(32,  128)(L)
        y = nn.Linear(128, 32*5*2)(y)
        
        # reshaping for decoder
        y = y.view(1, 32, 5, 2)
        
        # decoding
        x_hat = self.decoder(y)
        x_hat = x_hat[:, :, :92, :42]

        return x_hat
    
    def forward(self, x):
        L     = self.forward_encoder(x)
        x_hat = self.forward_decoder(L)
        return x_hat

# Optimization definition
num_epochs = 20
batch_size = 100
learning_rate = 2e-3
    
    
model = audio_autoencoder()
#print(model)

criterion = nn.functional.mse_loss
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)

#%% Optimization run
writer = SummaryWriter()

for epoch in range(num_epochs):
    for data, label in data_loader:
        spec = data
        # ===== forward  =====
        output = model(spec)
        #print("Sizes:", spec.size(), output.size())
        loss   = criterion(output, spec)

        # ===== backward =====
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # ===== log =====
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1,
                                              num_epochs,
                                              loss.data.item()))
    img = output.squeeze(1)
    writer.add_image('epoch'+str(epoch), img)
    
    plt.figure(figsize=(10,5))
    plt.imshow(output[0,0,:,:].detach().numpy(), origin='lower')
    plt.show()

writer.close()    
#%%
torch.save(model.state_dict(), './models/audio_AE.pth')