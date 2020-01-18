# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:22:07 2020

@author: Hugo
"""
import numpy as np
import utils 
import loader as load
import datasets as data
import autoencod as ae

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

def AudioRetrieval(y):
    # Building the MIDI latent space
    dataset,labels = data.MIDIDataset()
    Slist = []
    Encoder = ae.midi_encoder()
    for snip in dataset :       
        L = Encoder.forward(snip)
        Slist.append(L)
    S = tuple(Slist)
    #Find the nearest MIDI files to y in the latent space
    NNindex = NnQuery(y,S)
    RetrievedMIDI = dataset[NNindex,:,:,:]
    return RetrievedMIDI

def MIDIRetrieval(y):
    # Building the MIDI latent space
    dataset,labels = data.Snippets()
    Slist = []
    Encoder = ae.audio_encoder()
    for snip in dataset :       
        L = Encoder.forward(snip)
        Slist.append(L)
    S = tuple(Slist)
    #Find the nearest MIDI files to y in the latent space
    NNindex = NnQuery(y,S)
    RetrievedAudio = dataset[NNindex,:,:,:]
    return RetrievedAudio
    
    
    
    
        