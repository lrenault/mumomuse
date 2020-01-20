# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:20:03 2020

@author: Hugo
"""
import retrevial as r
import pylab as plt

def TestSimple(y,S):
    """
    Simple Test for NNQuery in 2 dimensions
    
    """
    
    NNindex = r.NnQuery(y,S)
    plt.figure()
    plt.scatter(y[0],y[1],c = "coral")
    plt.scatter(S[0][:],S[1][:], c = "lightblue")
    plt.scatter(S[0][NNindex],S[1][NNindex], c = "green")
    plt.axis((-1,3,-1,3),option = "equal")
    
    print(r.CosDist(y,S[:][0]))
    print(r.CosDist(y,S[:][1]))
    return None

y = (1,0)

S1 = ((1,0),(0,2))
TestSimple(y,S1)
S2 = ((1,2),(2,2))
TestSimple(y,S2)
