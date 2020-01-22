import torch
import torchaudio
from numpy import pad
from torchvision import transforms
import utils
import loader
import argparse

#%% preprocessing definition
def get_PianoRoll(midi, stack=True):
        """
        Args:
            - midi (pretty_midi) : midi data.
            - stack (bool) : if True, stack all insturment into 1 piano roll.
        Out:
            data (C, 128, L_audio) : piano roll of the midi file.
        """
        nb_instru = 0
        length = 0
        fs = utils.sampling_period_from_length(midi.get_end_time())
        for instrument in midi.instruments:
            if not instrument.is_drum:
                instruRoll = instrument.get_piano_roll(fs=fs)
                instruLen = instruRoll.shape[1]
                if nb_instru == 0:              # init
                    if stack:
                        data = instruRoll
                        length = instruLen
                    else:
                        data = [instruRoll]
                else :
                    if stack:
                        if instruLen > length:  # instrument score longer than whole track length
                            data = pad(
                                    data,
                                    ((0, 0), (0, instruLen - length))
                                    )
                            length = instruLen
                            
                        if instruLen < length:  # instrument score shorter than whole track length
                            instruRoll = pad(
                                    instruRoll,
                                    ((0, 0), (0, length - instruLen))
                                    )
                        data += instruRoll
                    else:
                        data.append(instruRoll)
                nb_instru += 1
        # tensor conversion
        data = torch.Tensor(data)
        if stack:
            data = data.unsqueeze(0)            # add channel dimension
        return data

def midi_preproc(stack=True):
    """ Composition of transformations to apply for midi pre-processing.
    Arg:
        - stack (Boolean): stack all instrument into a 1-channel tensor or not.
    Out:
        - midi_preproc (transforms.Compose): midi preprocessing.
    """
    midi_preproc = transforms.Compose([
                lambda x: get_PianoRoll(x, stack=stack),
                ])
    return midi_preproc
    

def audio_preproc(target_sr):
    """ Composition of transformations to apply for audio pre-processing.
    Arg:
        - target_sr (int): target sample rate.
    Out:
        - preproc (transforms.Compose): audio preprocessing.
    """
    audio_preproc = transforms.Compose([
                lambda x: torchaudio.transforms.Resample(x[1],
                                                         target_sr)(x[0]),
                transforms.Lambda(lambda x: x.mean(0).unsqueeze(0)), # convert to mono
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=target_sr,
                    n_fft=2048,
                    win_length=2048,
                    f_min=30.0,
                    f_max=6000.0,
                    n_mels=92),
                transforms.Normalize([0], [1]),
                ])
    return audio_preproc

#%% main
def main(raw_midi_path='db/nottingham-dataset-master/MIDI',
         raw_audio_path='db/nottingham-dataset-master/AUDIO',
         midi_snip_export_path='db/splitMIDI',
         audio_snip_export_path='db/splitAUDIO'
         ):
    """Main preprocessing function. Convert raw audio and midi files into
    splitted tensors and export them in given export paths.
    Args:
        - raw_midi_path (path): path to the folder containing the MIDI dataset.
        - raw_audio_path (path): path to the folder containing the audio dataset.
        - midi_snip_export_path (path): export path for the splitted midi snippet dataset.
        - audio_snip_export_path (path): export path for the splitted audio snippet dataset.
    """
    midi_loader = loader.MIDILoader()
    audio_loader = loader.AudioLoader()
    
    midi_loader.split_and_export_dataset(raw_midi_path)
    audio_loader.split_and_export_dataset(raw_audio_path)
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the given audio and midi datasets.')
    
    parser.add_argument('-midi', '--rawMIDI',
                        default='db/nottingham-dataset-master/MIDI',
                        type=str, help='Path to the raw midi dataset.',
                        dest='midiPATH')
    parser.add_argument('-audio', '--rawAUDIO',
                        default='db/nottingham-dataset-master/AUDIO',
                        type=str, help='Path to the raw audio dataset.',
                        dest='audioPATH')
    parser.add_argument('-midiTo', '--MIDIexport',
                        default='db/splitMIDI',
                        type=str, help='MIDI snippets export folder.',
                        dest='midiTo')
    parser.add_argument('-audioTo', '--AUDIOexport',
                        default='db/splitAUDIO',
                        type=str, help='Audio snippets export folder.',
                        dest='audioTo')
    
    options = parser.parse_args()
    
    main(options.midiPATH, options.audioPATH, options.midiTo, options.audioTo)
