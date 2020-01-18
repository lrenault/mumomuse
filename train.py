import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optparse import OptionParser
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
import loader
import autoencod
import utils

# CUDA for Pytorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#%% train and test definition
def train_AE(model, MODE, train_loader, optimizer, criterion, epoch):
    """ Training method for auto-encoders.
    Args :
        - model (nn.Module) : autoencoder model to train.
        - mode (MIDI_AE or AUDIO_AE) : midi or train auto-encoder train.
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


def eval_AE(model, MODE, test_loader, criterion, writer, epoch):
    """ Testing method for autoencoders.
    Args :
        - model (nn.Module) : autoencoder model to test.
        - mode (MIDI_AE or AUDIO_AE) : midi or train auto-encoder train.
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
#%% whole main process
def main(DatasetPATH='nottingham-dataset-master',
         MODE='MUMOMUSE',
         pretrained_model=None,
         dataset_reduced_to=None,
         num_epochs=15):
    """ Main training function.
    Args:
        - DatasetPATH (str): name of the dataset containing MIDI and AUDIO raw files folder.
        - MODE ('MUMOMUSE', 'MIDI_AE', 'AUDIO_AE'): train multimodal model, midi auto-encoder, or audio auto-encoder.
        - pretrained_model (path): resume training from the given model state.
        - dataset_reduced_to (int): reduce the dataset for small training. None for no dataset reduction.
    """
    dataset_reduced_to = 320 
    MODE = 'MUMOMUSE' # 'AUDIO_AE', 'MIDI_AE'

    # Snippet Dataset construction 
    midi_dataset  = datasets.Snippets('db/splitMIDI')
    audio_dataset = datasets.Snippets('db/splitAUDIO')

    midi_loader  = loader.MIDILoader()
    audio_loader = loader.AudioLoader()

    # generate snippets from raw data if snippet folder is empty
    if len(midi_dataset) == 0:
        MIDIpath = 'db/' + DatasetPATH + '/MIDI'
        midi_loader.split_and_export_dataset(MIDIpath)
        midi_dataset = datasets.Snippets('db/splitMIDI')
        
    if len(audio_dataset) == 0:
        audiopath = 'db/' + DatasetPATH + '/AUDIO'
        audio_loader.split_and_export_dataset(audiopath)
        audio_dataset = datasets.Snippets('db/splitAUDIO')
        
    pairs_dataset = datasets.PairSnippets(midi_dataset, audio_dataset)

    # Train-validation-test split
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
    
    
    train_loader = DataLoader(train_set, batch_size=30)
    valid_loader = DataLoader(valid_set, batch_size=1)
    test_loader  = DataLoader(test_set,  batch_size=1)

    # Optimization definition
    
    
    if MODE == 'MUMOMUSE':
        model = autoencod.multimodal()
        if pretrained_model:
            model.load_state_dict(torch.load(pretrained_model))
        criterion = utils.pairwise_ranking_objective()
    else:
        if MODE == 'MIDI_AE':
            model = autoencod.midi_AE()
        else: # 'AUDIO_AE'
            model = autoencod.audio_AE()
        criterion = nn.functional.mse_loss
    
    # optimization definitinon
    learning_rate = 2e-3
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)

    # Training
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('epoch [{}], training...'.format(epoch+1))
        
        if MODE == 'MUMOMUSE':
            train_multimodal(model, train_loader, optimizer, criterion, epoch)
            print('epoch [{}], end of training. Now evaluating...'.format(
                    epoch+1))
            train_loss = eval_multimodal(model, train_loader, criterion, 
                                         epoch, writer, "Train")
            test_loss  = eval_multimodal(model, valid_loader, criterion,
                                         epoch, writer, "Validation")
            print('epoch [{}]: train loss: {:.4f}, evaluation loss: {:.4f}'.format(
                    epoch+1, train_loss.data.item(), test_loss.data.item()))
            
        else: # autoencoder train
            train_AE(model, MODE, train_loader, criterion, optimizer, epoch)
            print('epoch [{}], end of training.'.format(epoch+1))
            loss = eval_AE(model, MODE, test_loader, criterion, writer, epoch)
            print('epoch [{}], validation loss: {:.4f}'.format(epoch+1, loss.data.item()))
    
    # Testing
    print("Now testing...")
    if MODE == 'MUMOMUSE':
        loss = eval_multimodal(model, test_loader, criterion, 0, writer, "Test")
    else:
        loss = eval_AE(model, MODE, test_loader, criterion, writer, 0)
    print("Test loss:", loss)
    writer.add_scalar("Test loss", loss, 0)
    
    writer.close()   
    
    # save model
    if MODE == 'MUMOMUSE':
        torch.save(model.state_dict(), './models/multimodal_small3.pth')
    elif MODE == 'MIDI_AE':
        torch.save(model.state_dict(), './models/midi_AE2.pth')
    else: # 'AUDIO_AE'
        torch.save(model.state_dict(), './models/audio_AE3.pth')
#%%
main()
#%% main call
if __name__ == "__main__":
    
	parser = OptionParser("usage: %prog [options] <path to database>")

	parser.add_option("-e", "--epochs", type="int",
	                  help="Number of Epochs",
	                  dest="epochs", default=20)

	parser.add_option("-g", "--gpu", type="int",
	                  help="ID of the GPU, run in CPU by default.", 
	                  dest="gpu")

	parser.add_option("-o", "--outPath", type="string",
	                  help="Path for the temporary folder.", 
	                  dest="outPath", default="OUT/")

	parser.add_option("-l", "--learning_rate", type="float",
	                  help="Value of the starting learning rate", 
	                  dest="learning_rate", default=1e-2)

	options, arguments = parser.parse_args()
	
	if len(arguments) == 1:
		main(arguments[0], EPOCHS=options.epochs, gpu=options.gpu,outPath=options.outPath, learning_rate=options.learning_rate)

	else:
		parser.error("You have to specify the path of the database.")