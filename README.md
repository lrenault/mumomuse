# mumomuse

[ATIAM 2019 ML Project] Implementation of the article "Learning audio-sheet music correspondences for cross-modal retrieval and piece identification" by M. Dorfer &amp; al., using MIDI files for the symbolic representation.

## Getting Started

### Dependencies

Python libraries used for running this project:

* pytorch
* [torchaudio](https://github.com/pytorch/audio) (v0.4.0)
* [pretty_midi](https://github.com/craffel/pretty-midi) (0.2.8)
* tensorboard 


### Set-up for training

Create a `/db/` folder in the working folder containing:
* the audio and matching MIDI datasets.
* a `/splitAUDIO/` folder for containing the audio snippets.
* a `/splitMIDI/` folder for keeping the midi snippets.

Then run:

```
train.py
```

The embedded space construction can be viewed via Tensorboard using the command:
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
example
```

Giving a complete wav file, generate the MIDI associated using the given MIDI snippets dataset as base.
```
example
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
