import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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

pairs_dataset = datasets.PairSnippets(midi_dataset, audio_dataset)

#%% Train-validation-test split
set_size = len(pairs_dataset)

# dataset reduction
if dataset_reduced_to:
    used_dataset, leftlovers = torch.utils.data.random_split(
            pairs_dataset,
            [dataset_reduced_to, set_size - dataset_reduced_to])
    set_size = dataset_reduced_to

# train-test split
train_size = int(set_size * 0.8)

train_set, test_set = torch.utils.data.random_split(
        used_dataset,
        [train_size, set_size - train_size])

# train-validation split
valid_size = int(train_size * 0.05)
train_set, valid_set = torch.utils.data.random_split(
        train_set,
        [train_size - valid_size, valid_size])


train_loader = DataLoader(train_set, batch_size=10)
valid_loader = DataLoader(valid_set, batch_size=1)
test_loader  = DataLoader(test_set,  batch_size=1)


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
    k = 0
    for midi, audio, label in train_loader:
        if MODE == 'MIDI_AE':
            data = midi.to(device)
        else:
            data = audio.to(device)
        # forward
        output = model(data)
        loss   = criterion(output, data)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # log
        if k%100==0:
            print('Trained with', k,
                  'snippets. Current loss :', loss.data.item())
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
        for midi, audio, label in test_loader:
            if MODE == 'MIDI_AE':
                data = midi.to(device)
            else:
                data = audio.to(device)
            # encode
            output = model(data)
            test_loss += criterion(output, data).data.item()
            
    test_loss /= len(test_loader.dataset)
    writer.add_scalar('Test loss', test_loss, epoch)
    
    target = data.squeeze(0)
    img = output.squeeze(0)
    writer.add_image('epoch' + str(epoch) + ' original', target, epoch)
    writer.add_image('epoch' + str(epoch) + ' reconstructed', img, epoch)
    
    # instant log
    plt.imshow(target.squeeze(0))
    plt.title('Target')
    plt.show()
    plt.imshow(img.squeeze(0))
    plt.title('Reconstructed')
    plt.show()
    
    return test_loss



def train_multimodal(model, train_loader, optimizer, criterion, epoch):
    """ Training method for multimodal network.
    Args :
        - model (nn.Module) : model to train.
        - train_loader (DataLoader) : Loader of the matching pairs training set.
        - optimizer (optim) : optimization method.
        - criterion (nn.Module) : loss function.
        - epoch (int) : training iteration number.
    """
    model.train()
    k = 0
    for batch_midi, batch_audio, batch_labels in train_loader:
        try:
            # batch generation
            anti_audios = utils.batch_except(train_loader.dataset,
                                             batch_labels, 49)
            # to device
            batch_midi  = batch_midi.to(device)
            batch_audio = batch_audio.to(device)
            anti_audios = anti_audios.to(device)
            
            # forward
            emb_midi, emb_audio = model(batch_midi, batch_audio)
            emb_anti_audios = model.g(anti_audios)
            # compute loss
            loss = criterion(emb_midi, emb_audio, emb_anti_audios)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            k += 1
            
        except FileNotFoundError:
            print('FileNotFoundError')
            pass
        except KeyError:
            print('KeyError')
            pass
        
        if k%5 == 0:
            print("Trained with", k, "snippets.")
    return None

def eval_multimodal(model, loader, criterion, epoch, writer, set_name):
    """ Testing method for multimodal network.
    Args :
        - model (nn.Module) : model to train.
        - loader (DataLoader) : Loader of the matching pairs test/validation set.
        - criterion (nn.Module) : loss function.
        - epoch (int) : training iteration number.
        - writer (Summarywriter) : for datalog.
        - set_name (string) : 
    Output :
        - test_loss (loss) : computed loss value.
    """
    model.eval()
    loss = 0
    with torch.no_grad():
        midi_mat  = torch.zeros(1, 32)
        audio_mat = torch.zeros(1, 32)
        midi_metadata  = ['None']
        audio_metadata = ['None']
        for batch_midi, batch_audio, batch_labels in loader:
            try:
                # batch generation
                anti_audios = utils.batch_except(loader.dataset,
                                                 batch_labels, 49)
                # to device
                batch_midi  = batch_midi.to(device)
                batch_audio = batch_audio.to(device)
                anti_audios = anti_audios.to(device)
                
                # encode
                emb_midi, emb_audio = model(batch_midi, batch_audio)
                emb_anti_audios = model.g(anti_audios)

                # compute loss
                loss += criterion(emb_midi, emb_audio, emb_anti_audios)
                
                # add to metadata
                midi_mat = torch.cat((midi_mat, emb_midi), 0)
                audio_mat = torch.cat((audio_mat, emb_audio), 0)
                
                for label in batch_labels:
                    midi_metadata.append( label + '_midi')
                    audio_metadata.append(label + '_audio')
            
            except FileNotFoundError:
                print('FileNotFoundError')
                pass
            except KeyError:
                print('KeyError')
                pass
        
        mat = torch.cat((midi_mat, audio_mat), 0)
        metadata = midi_metadata + audio_metadata
        
        writer.add_embedding(mat, metadata=metadata, global_step=epoch)
            
    loss /= len(loader.dataset)
    writer.add_scalar(set_name + ' loss', loss, epoch)
    
    return loss

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
        train_multimodal(model, train_loader, optimizer, criterion, epoch)
        print('epoch [{}], end of training. Now evaluating...'.format(
                epoch+1))
        train_loss = eval_multimodal(model, train_loader, criterion, 
                                     epoch, writer, "Train")
        test_loss  = eval_multimodal(model, test_loader, criterion,
                                     epoch, writer, "Test")
        print('epoch [{}]: train loss: {:.4f}, test loss: {:.4f}'.format(
                epoch+1, train_loss.data.item(), test_loss.data.item()))

    else:
        print('epoch [{}], training...'.format(epoch+1))
        train_AE(model, train_loader, criterion, optimizer, epoch)
        print('epoch [{}], end of training.'.format(epoch+1))
        loss = test_AE(model, test_loader, criterion, writer, epoch)
        print('epoch [{}], test loss: {:.4f}'.format(epoch+1, loss.data.item()))

writer.close()   
#%% save model
if MODE == 'MUMOMUSE':
    torch.save(model.state_dict(), './models/multimodal_small3.pth')
elif MODE == 'MIDI_AE':
    torch.save(model.state_dict(), './models/midi_AE2.pth')
else: # 'AUDIO_AE'
    torch.save(model.state_dict(), './models/audio_AE3.pth')
    
#%% Test
model.load_state_dict(torch.load('models/multimodal_small3.pth'))
writer = SummaryWriter()
loss = eval_multimodal(model, train_loader, criterion, 0, writer, "Train")
writer.close()