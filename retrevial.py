# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:22:07 2020

@author: Hugo
"""
import numpy as np
import utils 
import datasets as data
import autoencod as ae
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

def AudioRetrieval(x,model_dir = "models/multimodal_small.pth", data_dir = "db/splitMIDI"):
    Encoder = ae.multimodal()
    Encoder.load_state_dict(load(model_dir))
    # Put the spectrogram x in the latent space
    _,y = Encoder.forward(x) 
    # Sets the MIDI points in the latent space
    dataset,labels = data.Snippets(data_dir)
    Slist = []
    for snip in dataset :       
        L,_ = Encoder.forward(snip)
        Slist.append(L)
    S = tuple(Slist)
    #Find the nearest MIDI files to y in the latent space
    NNindex = NnQuery(y,S)
    RetrievedMIDI = dataset[NNindex,:,:,:]
    print(labels[NNindex])
    return RetrievedMIDI

def MIDIRetrieval(x,model_dir = "models/multimodal_small.pth",data_dir = "db/SplitAudio"):
    Encoder = ae.multimodal()
    Encoder.load_state_dict(load(model_dir))
    #Put the MIDI snippet x in the latent space
    y,_ = Encoder.forward(x)
    # Sets the Audio points in the latent space
    dataset,labels = data.Snippets(data_dir)
    Slist = []
    for snip in dataset :       
        _,L = Encoder.forward(snip)
        Slist.append(L)
    S = tuple(Slist)
    #Finds the nearest MIDI files to y in the latent space
    NNindex = NnQuery(y,S)
    RetrievedAudio = dataset[NNindex,:,:,:]
    print(labels[NNindex])
    return RetrievedAudio
    
    
    
    
        