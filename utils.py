import torch
import torch.nn as nn
from torch.nn.functional import normalize
from numpy import log
from random import choice

def sampling_period_from_length(end_time):
    """Empirical affine function for giving an adapted sampling period given the music length.
    Arg:
        - end_time (float) : length of the pretty_midi file
    Out:
        - Ts (float) : sampling period for the piano roll.
        """
    a = (21.9 - 23.3) / (log(150) - log(20))
    b = 23.3 - a * log(20)
    Ts = a * log(end_time) + b
    return Ts

def previous_label(label):
    """ Give previous snippet label of the given snippet label.
    Arg:
        - label (string): current snippet label.
    Out:
        - previous (string): label of the previous label.
    """
    root, snip_idx = label.split('_')
    prev_snip_idxs = str(int(snip_idx) - 1)
    previous = root + '_' + prev_snip_idxs
    return previous

def batch_except(dataset, excepts, batch_size=99):
    """Draw a random list of audio snippets in the given dataset, excepts the excepts.
    Args :
        - dataset (PairSnippets) : dataset to draw audio snippets from.
        - excepts (list) : list of labels not to draw.
        - batch_size (int) : length of the output batch.
    Outs :
        - batch (batch_size, audio_snippet.size()) : random batch without excepts.
    """ 
    batch = []
    if batch_size < (len(dataset) - len(excepts)) // 3 : # avoid too many draws
        for i in range (batch_size):
            candidate = choice(dataset)
            while candidate[2] in excepts:
                candidate = choice(dataset)
            batch.append(candidate[1])
    else:
        for midi, audio, label in dataset:
            if label not in excepts:
                batch.append(audio)
                if len(batch) > batch_size:
                    break
    return torch.stack(batch)
    
def s(x, y):
    """ Cosine similarity between two tensors.
    Args:
        - x, y (N, D): embedded vectors.
    """
    return nn.CosineSimilarity()(normalize(x), normalize(y))

class pairwise_ranking_objective(nn.Module):
    """Hinge loss"""
    def __init__(self, margin=0.7):
        """
        Args :
            - margin : margin of the loss function.
        """
        super(pairwise_ranking_objective, self).__init__()
        self.margin = margin
    
    def forward(self, midi_match, audio_match, contrastive_audios):
        """
        Args:
            - midi_match (1, 32) : embedding of a midi excerpt.
            - audio_match (1, 32) : embedding of its matching audio excerpt.
            - contrastive_audio (99, 32) : embedding of contrastive audio excerpts.
        """
        loss = 0
        for audio in torch.split(contrastive_audios, 1):
            loss += max(0, self.margin \
                            - torch.sum(s(midi_match, audio_match)) \
                            + torch.sum(s(midi_match, audio)))
        return loss    