# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:22:07 2020

@author: Hugo
"""
import numpy as np
import utils 
import datasets as data
import autoencod as ae
import torch.nn.Module as mod
from torch import load

def NnQuery(y,S):
    """
    Finds the nearest neighbour of y in the space S according to the cosine distance
    y = Numpy vector 
    S = Numpy Array
    """
    
    iMax = np.shape(S)[1]
    
    d = np.zeros((2,iMax))
    for i in range(iMax):
        d[0][i] = i
        d[1][i] = 1.0 - utils.s(y,S[:][i])
        
    dSorted = np.sort(d)
    NNindex = np.int32(dSorted[0][0])
    
    return NNindex

def AudioRetrieval(y,model_dir = "models/multimodal_small.pth"):
    # Sets the MIDI points in the latent space
    dataset,labels = data.MIDIDataset()
    Slist = []
    Encoder = ae.multimodal()
    mod.load_state_dict(load(model_dir))
    for snip in dataset :       
        L,_ = Encoder.forward(snip)
        Slist.append(L)
    S = tuple(Slist)
    #Find the nearest MIDI files to y in the latent space
    NNindex = NnQuery(y,S)
    RetrievedMIDI = dataset[NNindex,:,:,:]
    return RetrievedMIDI

def MIDIRetrieval(y,model_dir = "models/multimodal_small.pth"):
    # Sets the MIDI points in the latent space
    dataset,labels = data.Snippets()
    Slist = []
    Encoder = ae.multimodal()
    mod.load_state_dict(load(model_dir))
    for snip in dataset :       
        _,L = Encoder.forward(snip)
        Slist.append(L)
    S = tuple(Slist)
    #Finds the nearest MIDI files to y in the latent space
    NNindex = NnQuery(y,S)
    RetrievedAudio = dataset[NNindex,:,:,:]
    return RetrievedAudio
    
    
    
    
        