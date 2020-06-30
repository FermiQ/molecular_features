"""
Created on Mon Apr 27 11:45:32 2020

@author: Patrick
"""

import numpy as np
from molreps.methods import rigid_transform

# Random rotation and translation
R = np.random.rand(3,3)
t = np.random.rand(3,1)

# make R a proper rotation matrix, force orthonormal
U, S, Vt = np.linalg.svd(R)
R = np.dot(U,Vt)

# remove reflection
if np.linalg.det(R) < 0:
   Vt[2,:] *= -1
   R = np.dot(U,Vt)
print("det:",np.linalg.det(R))

# number of points


n = 10

A = np.random.rand(3, n)
B = np.dot(R,A) + np.tile(t, (1, n))
B= B.T
A = A.T

# Recover R and t
B2,ret_R, ret_t = rigid_transform(A, B)

# Compare the recovered R and t with the original


# Find the root mean squared error
err = B2 - B
err = np.multiply(err, err)
err = np.sum(err)
rmse = np.sqrt(err/n);

print("Points A")
print(A)
print("")

print("Points B")
print(B)
print("")

print("Ground truth rotation")
print(R)

print("Recovered rotation")
print(ret_R)
print("")

print("Ground truth translation")
print(t)

print("Recovered translation")
print(ret_t)
print("")


print("RMSE:", rmse)

if rmse < 1e-5:
    print("Everything looks good!\n");
else:
    print("Hmm something doesn't look right ...\n");
