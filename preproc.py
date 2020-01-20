import argparse
import loader

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
