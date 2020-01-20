# mumomuse

[ATIAM 2019 ML Project] Implementation of the article "Learning audio-sheet music correspondences for cross-modal retrieval and piece identification" by M. Dorfer &amp; al., using MIDI files for the symbolic representation.

## Getting Started

The dependencies and directories can be installed using the command:
```
sh init.sh
```

### Dependencies

Python libraries used for running this project:

* pytorch
* [torchaudio](https://github.com/pytorch/audio) (v0.4.0)
* [pretty_midi](https://github.com/craffel/pretty-midi) (0.2.8)
* tensorboard

Run `hello.py` to monitor which package is missing.

### Folder organization

The directories should be organized as follow:
```
mumomuse
└──db
    └── yourDataset
        └── AUDIO
            └── *.wav
        └── MIDI
            └── *.mid
    └── splitMIDI
    └── splitAUDIO
```

## Data preprocessing

Preprocess the raw audio and midi files in the given datasets into splitted snippet tensors, then the snippets are stored in the given export folder.

```
usage: preproc.py [-h] [-midi MIDIPATH] [-audio AUDIOPATH] [-midiTo MIDITO]
                  [-audioTo AUDIOTO]

Preprocess the given audio and midi datasets.

optional arguments:
  -h, --help            show this help message and exit
  -midi MIDIPATH, --rawMIDI MIDIPATH
                        Path to the raw midi dataset.
  -audio AUDIOPATH, --rawAUDIO AUDIOPATH
                        Path to the raw audio dataset.
  -midiTo MIDITO, --MIDIexport MIDITO
                        MIDI snippets export folder.
  -audioTo AUDIOTO, --AUDIOexport AUDIOTO
                        Audio snippets export folder.
```

## How to train your network

Train a multimodal, audio auto-encoder or midi auto-encoder model using the audio and midi snippets datasets.

```
usage: train.py [-h] [-midi MIDIPATH] [-audio AUDIOPATH] [-gpu GPU_ID]
                [-r REDUCE] [-t PRETRAINED] [-m MODE] [-e NUM_EPOCHS]
                [-b BATCH]

Train a model with given dataset.

optional arguments:
  -h, --help            show this help message and exit
  -midi MIDIPATH, --midiPATH MIDIPATH
                        Folder containing the midi snippets (db/splitMIDI by default).
  -audio AUDIOPATH, --audioPATH AUDIOPATH
                        Folder containing the audio snippets (db/splitAUDIO by default).
  -gpu GPU_ID           GPU ID.
  -r REDUCE, --reduce REDUCE
                        Dataset length reduction to. (None by default).
  -t PRETRAINED, --trained PRETRAINED
                        Pretrained model path.
  -m MODE, --mode MODE  Model type to be trained ("MUMOMUSE", "AUDIO_AE" or
                        "MIDI_AE").
  -e NUM_EPOCHS, --epochs NUM_EPOCHS
                        Number of epochs.
  -b BATCH, --batch BATCH
                        Batch size.
```

The embedded space construction can be viewed via Tensorboard using the command:
```
tensorboard --logdir=runs
```

Our models' training process can be viewed using the command:

```
tensorboard --logdir=valid_runs
```

### Two-way snippets retrieval

Benchmark the model's retrieval performances using the snippets in `db/splitAUDIO` and `db/splitMIDI`:
```
retrevial.py
```

### Piece Identification and Performance Retrieval

Giving a complete MIDI file, generate the audio associated using the given audio snippets dataset as base.
```
TBA
```

Giving a complete wav file, generate the MIDI associated using the given MIDI snippets dataset as base.
```
TBA
```

## Contributing
List of contributors who participated in this project:
* [L. Renault](https://github.com/lrenault)
* [H. Fafin](https://github.com/hfafin)
* [B. Souchu](https://github.com/BrunoSouchu)
* [A. Terrasse](https://github.com/aterrasse)

## Authors

* **M. Dorfer &amp; al.** - *"Learning audio-sheet music correspondences for cross-modal retrieval and piece identification" (2018)* - [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.12/)

## License

This project is open-source.

## Acknowledgments

* Inspiration...
