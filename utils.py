import torch
import torch.nn as nn
from torch.nn.functional import normalize
from numpy import log

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

def random_except(set_size, excepts, idxs_set_size=99, method=1):
    """Draw a set of random exclusive natural integers that aren't in the given exception set.
    Args :
        - set_size (int) : range upper limit.
        - excepts (list) : list of integers not to draw.
        - idxs_set_size (int) : number of integers to draw.
    Outs :
        - idxs_set (list) : list of random indexes.
    """ 
    if method==1:
        idxs_set = torch.LongTensor(idxs_set_size).random_(set_size - 1)
        for i in range(idxs_set_size):
            if idxs_set[i] >= excepts[0]:
                idxs_set[i] += 1
                
    elif method==2:
        set_to_draw_from = list(range(set_size))
        for i in range(set_size - 1, 0, -1):
            if set_to_draw_from[i] in excepts:
                del set_to_draw_from[i]
        elements = torch.LongTensor(idxs_set_size).random_(len(set_to_draw_from))
        idxs_set = [set_to_draw_from[i] for i in elements]
        
    return idxs_set
    

def s(x, y):
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
            - midi_match (1, 1, 32) : embedding of a midi excerpt.
            - audio_match (1, 1, 32) : embedding of its matching audio excerpt.
            - contrastive_audio (99, 1, 32) : embedding of contrastive audio excerpts.
        """
        loss = 0
        for audio in contrastive_audios:
            loss += max(0, self.margin \
                            - s(midi_match, audio_match) \
                            + s(midi_match, audio))
        return loss
    