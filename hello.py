print("Hello world!")

try:
    import torch
    # CUDA for Pytorch
    use_cuda = torch.cuda.is_available()

    available = " " if use_cuda else " not "
    print("Cuda is" + available + "available.")

except ImportError:
    print("Please import pytorch")

try:
    import torchvision
    print("torchvision imported.")
except ImportError:
    print("torchvision not found.")

try:
    import pretty_midi
    print("pretty_midi imported.")
except ImportError:
    print("pretty_midi not found.")

try:
    import tensorboard
    print("tensorboard imported.")
except ImportError:
    print("tensorboard not found.")

try:
    import torchaudio
    print("pyaudio imported.")
except ImportError:
    print("torch audio not found.")