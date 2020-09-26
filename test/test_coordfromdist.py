# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:09:40 2020

@author: Patrick
"""

from molreps.methods.geo_npy import coordinates_to_distancematrix,coordinates_from_distancematrix
import matplotlib.pyplot as plt
import numpy as np

X= np.array([[0,0,0],[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],[-5,0,0]])
print(X)
dist = coordinates_to_distancematrix(X)
print(dist)
out =coordinates_from_distancematrix(dist,use_center =None)
out[np.abs(out)<1e-6] = 0
print(out)
print("Centered Coordinates",np.sum(out,axis=0))
check = coordinates_to_distancematrix(out)
print("Max error:")
print(np.amax(np.abs(check-dist)))