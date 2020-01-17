# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:22:07 2020

@author: Hugo
"""
import numpy as np

def CosDist(x,y):
    """
    Computes the Cosine Distance between two vectors 
    x,y : vectors
    """
    dist = 1-( np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)) )
    return  dist

def NnQuery(y,S,dist = "cosine"):
    """
    Finds the nearest neighbour of y in the space S according to the distance dist
    y = Numpy vector 
    S = Numpy Array
    dist = Distance function (Cosine by default)
    """
    iMax = np.shape(S)[1]
    d = np.zeros(2,iMax)
    for i in range(iMax):
        d[1][i] = i
        d[2][i] = CosDist(y,S[:][i])
        
    dSorted = np.sort(d)
    NNindex = dSorted[1][1]
    
    return NNindex
    
        