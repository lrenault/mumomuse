import torch.nn as nn

def s(x, y):
    return nn.CosineSimilarity()(x, y)

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
    