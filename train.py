import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import snippets
import autoencod
import utils

cpu = torch.device("cpu")

#%% train and test definition
def train_AE(model, MODE, train_loader, optimizer, criterion, epoch, device):
    """ Training method for auto-encoders.
    Args :
        - model (nn.Module): autoencoder model to train.
        - mode (MIDI_AE or AUDIO_AE): midi or train auto-encoder train.
        - train_loader (Dataloader): Dataloader of the train set.
        - optimizer (torch.optim): optimization method.
        - criterion (nn.Functional): loss function.
        - epoch (int): training iteration number.
        - device (cuda.device): cpu or gpu device.
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
        if (k % 100)==0:
            print('Trained with', k,
                  'snippets. Current loss :', loss.data.item())
    return None


def eval_AE(model, MODE, test_loader, criterion, writer, epoch, device):
    """ Testing method for autoencoders.
    Args :
        - model (nn.Module) : autoencoder model to test.
        - mode (MIDI_AE or AUDIO_AE) : midi or train auto-encoder train.
        - test_lodaer (DataLoader) : Dataloader of the test set.
        - criterion (nn.Functional) : loss function.
        - writer (SummaryWriter) : for data log.
        - epoch (int) : testing iteration number.
        - device (cuda.device): cpu or gpu device.
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
    writer.add_image('epoch' + str(epoch) + '_original', target, epoch)
    writer.add_image('epoch' + str(epoch) + '_reconstructed', img, epoch)

    return test_loss



def train_multimodal(model, train_loader, optimizer, criterion, epoch, device):
    """ Training method for multimodal network.
    Args :
        - model (nn.Module) : model to train.
        - train_loader (DataLoader) : Loader of the matching pairs training set.
        - optimizer (optim) : optimization method.
        - criterion (nn.Module) : loss function.
        - epoch (int) : training iteration number.
        - device (cuda.device): cpu or gpu device.
    """
    model.train()
    k = 0
    for batch_midi, batch_audio, batch_labels in train_loader:
        try:
            # batch generation
            anti_audios = utils.batch_except(train_loader.dataset,
                                             batch_labels, 99)
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
            
        except KeyError:
            print('KeyError')

        if k%10 == 0:
            print("Trained with", k, "snippet batches.")
    return None

def eval_multimodal(model, loader, criterion, epoch, writer, set_name, device):
    """ Testing method for multimodal network.
    Args :
        - model (nn.Module) : model to train.
        - loader (DataLoader) : Loader of the matching pairs test/validation set.
        - criterion (nn.Module) : loss function.
        - epoch (int) : training iteration number.
        - writer (Summarywriter) : for datalog.
        - set_name (string) : step label for tensorboard monitoring.
        - device (cuda.device): cpu or gpu device.
    Output :
        - test_loss (loss) : computed loss value.
    """
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        midi_mat  = torch.zeros(1, 32).to(device)
        audio_mat = torch.zeros(1, 32).to(device)
        midi_img  = torch.zeros(1, 1, 128, 42).to(device)
        audio_img = torch.zeros(1, 1, 92, 42).to(device)
        midi_metadata  = ['Zero_midi']
        audio_metadata = ['Zero_audio']
        for batch_midi, batch_audio, batch_labels in loader:
            try:
                # batch generation
                anti_audios = utils.batch_except(loader.dataset,
                                                 batch_labels, 99)
                # to device
                batch_midi  = batch_midi.to(device)
                batch_audio = batch_audio.to(device)
                anti_audios = anti_audios.to(device)

                # encode
                emb_midi, emb_audio = model(batch_midi, batch_audio)
                emb_anti_audios = model.g(anti_audios)

                # compute loss
                eval_loss += criterion(emb_midi, emb_audio, emb_anti_audios)

                # add to metadata
                midi_mat  = torch.cat((midi_mat, emb_midi), 0)
                audio_mat = torch.cat((audio_mat, emb_audio), 0)
                midi_img  = torch.cat((midi_img, batch_midi), 0)
                audio_img = torch.cat((audio_img, batch_audio), 0)

                for label in batch_labels:
                    midi_metadata.append(label + '_midi')
                    audio_metadata.append(label + '_audio')
                
            except KeyError:
                print('KeyError')
                pass
        
        # label images correction
        audio_img = nn.ZeroPad2d((43, 43, 18, 18))(audio_img)
        midi_img  = nn.ZeroPad2d((43, 43,  0,  0))(midi_img)
        
        midi_img = utils.toColor(midi_img)
        audio_img = utils.toColor(audio_img)
        
        mat = torch.cat((midi_mat, audio_mat), 0)
        img = torch.cat((midi_img, audio_img), 0)
        metadata = midi_metadata + audio_metadata
        
        # add to writer
        writer.add_embedding(mat, metadata=metadata, label_img=img,
                             global_step=epoch, tag=set_name)

    eval_loss /= len(loader.dataset)
    writer.add_scalar(set_name + ' loss', eval_loss, epoch)

    return eval_loss



#%% whole main process
def main(midi_snippets_path='db/splitMIDI',
         audio_snippets_path='db/splitAUDIO',
         GPU_ID='0',
         dataset_reduced_to=None,
         pretrained_model=None,
         MODE='MUMOMUSE',
         num_epochs=20,
         batch_size=1
         ):
    """ Main training function.
    Args:
        - midi_snippets_path (path): path to the folder containing the midi snippets.
        - audio_snippets_path (path): path to the folder containing the audio snippets.
        - GPU_ID (str) : ID of the GPU used for training acceleration.
        - MODE ('MUMOMUSE', 'MIDI_AE', 'AUDIO_AE'): train multimodal model, midi auto-encoder, or audio auto-encoder.
        - pretrained_model (path): resume training from the given model state.
        - dataset_reduced_to (int): reduce the dataset for small training. None for no dataset reduction.
    """
    # CUDA for Pytorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + GPU_ID if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    ### Snippet Dataset construction ###
    midi_dataset  = snippets.Snippets(midi_snippets_path)
    audio_dataset = snippets.Snippets(audio_snippets_path)

    # generate snippets from raw data if snippet folder is empty
    if len(midi_dataset) == 0:
        print("MIDI folder is empty of snippet!")
        raise

    if len(audio_dataset) == 0:
        print("Audio folder is empty of snippet!")
        raise

    pairs_dataset = snippets.PairSnippets(midi_dataset, audio_dataset)

    ### Train-validation-test split ###
    set_size = len(pairs_dataset)
    
    if set_size == 0:
        print("No common labels in audio and midi snippets folders!")
        raise

    # dataset reduction
    if dataset_reduced_to:
        pairs_dataset, leftlovers = torch.utils.data.random_split(
                pairs_dataset,
                [dataset_reduced_to, set_size - dataset_reduced_to])
        set_size = dataset_reduced_to

    # train-test split
    train_size = int(set_size * 0.8)

    train_set, test_set = torch.utils.data.random_split(
            pairs_dataset,
            [train_size, set_size - train_size])

    # train-validation split
    valid_size = int(train_size * 0.05)
    train_set, valid_set = torch.utils.data.random_split(
            train_set,
            [train_size - valid_size, valid_size])

    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size // 10 + 1)
    test_loader  = DataLoader(test_set,  batch_size=batch_size)

    # model definition
    if MODE == 'MUMOMUSE':
        model = autoencod.multimodal()
        criterion = utils.pairwise_ranking_objective(device=device)
    else:
        criterion = nn.functional.mse_loss
        if MODE == 'MIDI_AE':
            model = autoencod.midi_AE()
        else: # 'AUDIO_AE'
            model = autoencod.audio_AE()

    if pretrained_model:
        model.load_state_dict(torch.load(pretrained_model))
    if use_cuda:
        model.cuda(device=device)

    # optimization definitinon
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

    # Training
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('epoch [{}], training...'.format(epoch+1))
        
        if MODE == 'MUMOMUSE':
            train_multimodal(model, train_loader,
                             optimizer, criterion, epoch, device)
            print('epoch [{}], end of training. Now evaluating...'.format(
                    epoch+1))
            train_loss = 0
            train_loss = eval_multimodal(model, train_loader, criterion,
                                         epoch, writer, "Train", device)
            test_loss  = eval_multimodal(model, valid_loader, criterion,
                                         epoch, writer, "Validation", device)
            print('epoch [{}]: train loss: {:.4f}, validation loss: {:.4f}'.format(
                    epoch+1, train_loss.data.item(), test_loss.data.item()))
            
        else: # autoencoder train
            train_AE(model, MODE, train_loader, optimizer, criterion,
                     epoch, device)
            print('epoch [{}], end of training.'.format(epoch+1))
            
            loss = eval_AE(model, MODE, test_loader,
                           criterion, writer, epoch, device)
            print('epoch [{}], validation loss: {:.4f}'.format(epoch+1,
                  loss))
            
        torch.save(model.state_dict(), './temp/model_epoch' \
                                       + str(epoch) + '.pth')
        print('Model saved.')
        
    # Testing
    print("Now testing...")
    if MODE == 'MUMOMUSE':
        loss = eval_multimodal(model, test_loader, criterion,
                               0, writer, "Test", device)
    else:
        loss = eval_AE(model, MODE, test_loader, criterion, writer, 0, device)
    print("Test loss:", loss)
    writer.add_scalar("Test loss", loss, 0)

    writer.close()

    # save model
    if MODE == 'MUMOMUSE':
        torch.save(model.state_dict(), './models/multimodal_fat.pth')
    elif MODE == 'MIDI_AE':
        torch.save(model.state_dict(), './models/midi_AE2.pth')
    else: # 'AUDIO_AE'
        torch.save(model.state_dict(), './models/audio_AE3.pth')
    print("Final model saved.")


#%% main call
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with given dataset.')

    parser.add_argument('-midi', '--midiPATH', default='db/splitMIDI',
                        type=str,
                        help="Folder containing the midi snippets. Default='db/splitMIDI'",
                        dest='midiPATH')
    
    parser.add_argument('-audio', '--audioPATH', default='db/splitAUDIO',
                        type=str,
                        help="Folder containing the audio snippets. Default='db/splitAUDIO'",
                        dest='audioPATH')

    parser.add_argument('-gpu', default='0', type=str,
                        help="GPU ID. Default='0'", dest='GPU_ID')

    parser.add_argument('-r', '--reduce', default=None, type=int,
                        help='Dataset length reduction to. Default=None',
                        dest='reduce')

    parser.add_argument('-t', '--trained', default=None, type=str,
                        help='Pretrained model path. Default=None',
                        dest='pretrained')

    parser.add_argument('-m', '--mode',
                        default='MUMOMUSE', type=str,
                        help='Model type to be trained ("MUMOMUSE", "AUDIO_AE" or "MIDI_AE").',
                        dest='mode')

    parser.add_argument('-e', '--epochs', default=20, type=int,
                        help='Number of epochs. Default=20', dest='num_epochs')

    parser.add_argument('-b', '--batch', default=1, type=int,
                        help='Batch size. Default=1', dest='batch')

    options = parser.parse_args()

    print("MIDI snippets:", options.midiPATH,", Audio snippets:", options.audioPATH,
          "\n GPU ID:", options.GPU_ID,
          "\n mode:", options.mode,
          "\n pretrained:", options.pretrained,
          "\n reduction:", options.reduce,
          "\n nb epochs:", options.num_epochs,
          "\n batch size:", options.batch)

    main(midi_snippets_path=options.midiPATH,
         audio_snippets_path=options.audioPATH,
         GPU_ID=options.GPU_ID,
         dataset_reduced_to=options.reduce,
         pretrained_model=options.pretrained,
         MODE=options.mode,
         num_epochs=options.num_epochs,
         batch_size=options.batch
         )
