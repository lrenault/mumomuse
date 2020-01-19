print("Hello world!")

try:
    import torch
    # CUDA for Pytorch
    use_cuda = torch.cuda.is_available()

    available = "" if use_cuda else " not "
    print("Cuda is" + available + "available")

except ModuleNotFoundError:
    print("Please import pytorch")

try:
    import torchaudio
    print("pyaudio imported.")
except ModuleNotFoundError:
    print("torch audio not found.")

try:
    import torchvision
    print("torchvision imported.")
except ModuleNotFoundError:
    print("torchvision not found.")

try:
    import pretty_midi
    print("pretty_midi imported.")
except ModuleNotFoundError:
    print("pretty_midi not found.")

try:
    import tensorboard
    print("tensorboard imported.")
except ModuleNotFoundError:
    print("tensorboard not found.")
