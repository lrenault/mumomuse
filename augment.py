import torch
import numpy as np    
    
def addNoise(loader, augmented_proportion=0.1, noise_level=0.1):
    """
    Adds noise to a certain proportion of the snippet.
    Args :
        - augmented_proportion (float) = Proportion of snippets to add noise to.
        - noise_level (float) = Level of noise to be added.
        - loader (torch.utils.data.DataLoader) = Loader of the dataset to add noise to.
    """
    for snip,name in loader:
            
        augment = np.random.random() # use torch.random instead
        
        if(augment < augmented_proportion):  
            npSnip = snip.numpy()
            noise = np.random.random(size = npSnip.shape())
            noiseSnipNP = npSnip + noise
            snip = torch.from_numpy(noiseSnipNP)
            
    return None