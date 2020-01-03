import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import datasets
import loader
import autoencod
import utils

dataset_reduced_to = 320 # for small dataset. None for no dataset reduction
MODE = 'MUMOMUSE' # 'AUDIO_AE', 'MIDI_AE'

#%% Snippet Dataset construction 
midi_dataset  = datasets.Snippets('db/splitMIDI')
audio_dataset = datasets.Snippets('db/splitAUDIO')

midi_loader  = loader.MIDILoader
audio_loader = loader.AudioLoader

# generate snippets from raw data if snippet folder is empty
if len(midi_dataset) == 0:
    MIDIpath = 'db/nottingham-dataset-master/MIDI'
    midi_loader.split_and_export_dataset(MIDIpath)
    midi_dataset = datasets.Snippets('db/splitMIDI')

if len(audio_dataset) == 0:
    audiopath = 'db/nottingham-dataset-master/AUDIO'
    #audio_loader.split_and_export_dataset(audiopath)
    audio_dataset = datasets.Snippets('db/splitAUDIO')

#%% Train-test split
set_size = len(midi_dataset)

if dataset_reduced_to:
    midi_dataset, midi_leftlovers = torch.utils.data.random_split(
            midi_dataset,
            [dataset_reduced_to, set_size - dataset_reduced_to])
    set_size = dataset_reduced_to

train_size = int(set_size * 0.75)

midi_train_set, midi_test_set = torch.utils.data.random_split(
        midi_dataset,
        [train_size, set_size - train_size])

midi_snippet_train_loader = DataLoader(midi_train_set, num_workers=3)
midi_snippet_test_loader  = DataLoader(midi_test_set)

#%% Correspondance with audio dataset
music_names = audio_dataset.labels
idxs = list(range(len(audio_dataset)))

correspondance_dict = dict(zip(music_names, idxs))

if MODE == 'MUMOMUSE':
    pass

elif MODE == 'MIDI_AE':
    train_loader = midi_snippet_train_loader
    test_lodaer  = midi_snippet_test_loader
    
else: #'AUDIO_AE'
    train_indices = []
    test_indices  = []
    
    for snippet, label in midi_snippet_train_loader:
        train_indices.append(correspondance_dict(label))
    
    for snippet, label in midi_snippet_test_loader:
        test_indices.append(correspondance_dict(label))
        
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler  = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(audio_dataset, sampler=train_sampler)
    test_loader  = DataLoader(audio_dataset, sampler=test_sampler)

#%% train and test definition
def train_AE(model, train_loader, optimizer, criterion, epoch):
    """ Training method for auto-encoders.
    Args :
        - model (nn.Module) : autoencoder model to train.
        - train_loader (Dataloader) : Dataloader of the train set.
        - optimizer (torch.optim) : optimization method.
        - criterion (nn.Functional) : loss function.
        - epoch (int) : training iteration number.
    """
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
    return None


def test_AE(model, test_loader, writer, criterion, epoch):
    """ Testing method for autoencoders.
    Args :
        - model (nn.Module) : autoencoder model to test.
        - test_lodaer (DataLoader) : Dataloader of the test set.
        - writer (SummaryWriter) : for data log.
        - criterion (nn.Functional) : loss function.
        - epoch (int) : testing iteration number.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, label in test_loader:
            output = model(data)
            test_loss += criterion(output, data).data.item()
            
    test_loss /= len(test_loader.dataset)
    writer.add_scalar('Test loss', test_loss, epoch)
    
    target = data.squeeze(0)
    img = output.squeeze(0)
    writer.add_image('epoch' + str(epoch) + ' original', target)
    writer.add_image('epoch' + str(epoch) + ' reconstructed', img)
    
    # instant log
    plt.imshow(target.squeeze(0))
    plt.title('Target')
    plt.show()
    plt.imshow(img.squeeze(0))
    plt.title('Reconstructed')
    plt.show()
    
    return None

def train_multimodal():
    return None

def test_multimodal():
    return None

#%% Optimization definition
num_epochs = 15
learning_rate = 2e-3

if MODE == 'MUMOMUSE':
    model = autoencod.multimodal()
    criterion = utils.pairwise_ranking_objective()
else:
    if MODE == 'MIDI_AE':
        model = autoencod.midi_AE()
    else: # 'AUDIO_AE'
        model = autoencod.audio_AE()
    criterion = nn.functional.mse_loss

optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)
#%% Optimization run
writer = SummaryWriter()

for epoch in range(num_epochs):
    if MODE == 'MUMOMUSE':
        # TO BE COMPLETED
        #
        #
        #
        #
        #
        pass
    else:
        train_AE(model, train_loader, optimizer, epoch)
        print('epoch [{}], end of training.'.format(epoch+1))
        test_AE(model, test_loader, writer, epoch)
        print('epoch [{}], end of testing'.format(epoch+1))

writer.close()    
##%% save model
if MODE == 'MUMOMUSE':
    torch.save(model.state_dict(), './models/multimodal.pth')
elif MODE == 'MIDI_AE':
    torch.save(model.state_dict(), './models/midi_AE2.pth')
else: # 'AUDIO_AE'
    torch.save(model.state_dict(), './models/audio_AE3.pth')