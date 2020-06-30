"""
Created on Fri Apr 24 17:35:06 2020

@author: Patrick Reiser
"""

from molreps.methods import rotate_to_principle_axis
import matplotlib.pyplot as plt
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2], [0,0.5] , [0,0] , [-1,0.6]])
X= X+100

plt.scatter(X[:,0],X[:,1])
plt.show()

rotmat,Xr = rotate_to_principle_axis(X)
    
print(rotmat) 

plt.scatter(X[:,0],X[:,1])
plt.scatter(Xr[:,0],Xr[:,1])
plt.show()