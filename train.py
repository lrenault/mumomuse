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

# CUDA for Pytorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

#%% Snippet Dataset construction 
midi_dataset  = datasets.Snippets('db/splitMIDI')
audio_dataset = datasets.Snippets('db/splitAUDIO')

midi_loader  = loader.MIDILoader()
audio_loader = loader.AudioLoader()

# generate snippets from raw data if snippet folder is empty
if len(midi_dataset) == 0:
    MIDIpath = 'db/nottingham-dataset-master/MIDI'
    midi_loader.split_and_export_dataset(MIDIpath)
    midi_dataset = datasets.Snippets('db/splitMIDI')

if len(audio_dataset) == 0:
    audiopath = 'db/nottingham-dataset-master/AUDIO'
    audio_loader.split_and_export_dataset(audiopath)
    audio_dataset = datasets.Snippets('db/splitAUDIO')

#%% Correspondance with audio dataset
correspondance_dict = loader.get_dictionnary(midi_dataset, audio_dataset)

#%% Train-test-validation split
set_size = len(midi_dataset)

if dataset_reduced_to:
    midi_dataset, midi_leftlovers = torch.utils.data.random_split(
            midi_dataset,
            [dataset_reduced_to, set_size - dataset_reduced_to])
    set_size = dataset_reduced_to


train_size = int(set_size * 0.75)
non_train_size = set_size - train_size
midi_train_set, midi_non_training_set = torch.utils.data.random_split(
        midi_dataset,
        [train_size, non_train_size])


test_size = int(non_train_size * 0.25)
midi_test_set, midi_valid_set = torch.utils.data.random_split(
        midi_non_training_set,
        [test_size, non_train_size - test_size])


midi_snippet_train_loader = DataLoader(midi_train_set, num_workers=3)
midi_snippet_test_loader  = DataLoader(midi_test_set)
midi_snippet_valid_loader = DataLoader(midi_valid_set, num_workers=2)

#%% rename loaders for different Modes
if MODE == 'MUMOMUSE':
    pass

elif MODE == 'MIDI_AE':
    train_loader = midi_snippet_train_loader
    test_lodaer  = midi_snippet_test_loader
    
else: #'AUDIO_AE'
    train_indices = []
    test_indices  = []
    
    for snippet, label in midi_snippet_train_loader:
        train_indices.append(correspondance_dict[label])
    
    for snippet, label in midi_snippet_test_loader:
        test_indices.append(correspondance_dict[label])
        
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
        data, label = data.to(device), label.to(device)
        # forward
        output = model(data)
        loss   = criterion(output, data)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print('epoch [{}], loss:{:.4f}'.format(epoch+1, loss.data.item()))
    return None


def test_AE(model, test_loader, criterion, writer, epoch):
    """ Testing method for autoencoders.
    Args :
        - model (nn.Module) : autoencoder model to test.
        - test_lodaer (DataLoader) : Dataloader of the test set.
        - criterion (nn.Functional) : loss function.
        - writer (SummaryWriter) : for data log.
        - epoch (int) : testing iteration number.
    Output :
        - test_loss (loss) : computed loss value.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            # encode
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
    
    return test_loss



def train_multimodal(model, midi_train_loader, audio_dataset, correspondance,
                     optimizer, criterion, epoch):
    """ Training method for multimodal network.
    Args :
        - model (nn.Module) : model to train.
        - midi_train_loader (DataLoader) : Loader of the MIDI training set.
        - audio_dataset (Dataset) : Audio dataset.
        - correspondance (Dict) : correspondance dictionnary between audio and midi datasets.
        - optimizer (optim) : optimization method.
        - criterion (nn.Module) : loss function.
        - epoch (int) : training iteration number.
    """
    model.train()
    k = 0
    for batch_midi_snippets, batch_labels in midi_train_loader:
        try:
            # batch generation
            excepts = [correspondance_dict[label] for label in batch_labels]
            batch_idxs = utils.random_except(
                    len(audio_dataset),
                    excepts,
                    99)
            # forward
            emb_midi, emb_audio = model(
                    batch_midi_snippets,
                    audio_dataset[correspondance[batch_labels[0]]][0].unsqueeze(0)
                    )
            emb_anti_audios = [model.g(audio_dataset[idx][0].unsqueeze(0))
                                for idx in batch_idxs]
            # compute loss
            loss = criterion(emb_midi, emb_audio, emb_anti_audios)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            k += 1
            
        except FileNotFoundError:
            pass
        except KeyError:
            pass
        
        if k%5 == 0:
            print("Trained with", k, "snippets.")
    return None

def test_multimodal(model, midi_test_loader, audio_dataset, correspondance,
                    criterion, epoch, writer):
    """ Testing method for multimodal network.
    Args :
        - model (nn.Module) : model to train.
        - midi_train_loader (DataLoader) : Loader of the MIDI training set.
        - audio_dataset (Dataset) : Audio dataset.
        - correspondance (Dict) : correspondance dictionnary between audio and midi datasets.
        - criterion (nn.Module) : loss function.
        - epoch (int) : training iteration number.
        - writer (Summarywriter) : for datalog.
    Output :
        - test_loss (loss) : computed loss value.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        midi_mat = torch.zeros(1,32)
        audio_mat = torch.zeros(1,32)
        midi_metadata = ['None']
        audio_metadata = ['None']
        for midi_snippet, label in midi_test_loader:
            try:
                # batch generation
                excepts = [correspondance_dict[name] for name in label]
                batch_idxs = utils.random_except(len(audio_dataset), excepts, 99)
                # encode
                emb_midi, emb_audio = model(
                        midi_snippet,
                        audio_dataset[correspondance[label[0]]][0].unsqueeze(0)
                        )
                emb_anti_audio = [model.g(audio_dataset[idx][0].unsqueeze(0))
                                    for idx in batch_idxs]
                # compute loss
                test_loss += criterion(emb_midi, emb_audio, emb_anti_audio)
                
                # add to metadata
                midi_mat = torch.cat((midi_mat, emb_midi), 0)
                audio_mat = torch.cat((audio_mat, emb_audio), 0)
                
                midi_metadata.append(label[0] + '_midi')
                audio_metadata.append(label[0] + '_audio')
            
            except FileNotFoundError:
                print('FileNotFoundError')
                pass
            except KeyError:
                print('KeyError')
                pass
        
        mat = torch.cat((midi_mat, audio_mat), 0)
        metadata = midi_metadata + audio_metadata
        
        writer.add_embedding(mat, metadata=metadata, global_step=epoch)
            
    test_loss /= len(midi_test_loader.dataset)
    writer.add_scalar('Test loss', test_loss, epoch)
    
    return test_loss

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
        print('epoch [{}], training...'.format(epoch+1))
        train_multimodal(model, midi_snippet_train_loader, audio_dataset,
                         correspondance_dict,
                         optimizer, criterion, epoch)
        print('epoch [{}], end of training.'.format(epoch+1))
        
        loss = test_multimodal(model, midi_snippet_test_loader,
                               audio_dataset, correspondance_dict,
                               criterion, epoch, writer)
        print('epoch [{}], test loss:{:.4f}'.format(epoch+1, loss.data.item()))

    else:
        print('epoch [{}], training...'.format(epoch+1))
        train_AE(model, train_loader, criterion, optimizer, epoch)
        print('epoch [{}], end of training.'.format(epoch+1))
        loss = test_AE(model, test_loader, criterion, writer, epoch)
        print('epoch [{}], test loss:{:.4f}'.format(epoch+1, loss.data.item()))

writer.close()   
#%% save model
if MODE == 'MUMOMUSE':
    torch.save(model.state_dict(), './models/multimodal_small.pth')
elif MODE == 'MIDI_AE':
    torch.save(model.state_dict(), './models/midi_AE2.pth')
else: # 'AUDIO_AE'
    torch.save(model.state_dict(), './models/audio_AE3.pth')
    
#%% Validation
writer = SummaryWriter()
loss = test_multimodal(model, midi_snippet_valid_loader,
                       audio_dataset, correspondance_dict,
                       criterion, 0, writer)
writer.close()