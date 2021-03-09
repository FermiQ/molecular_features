"""Old unused functions.

Do not use! For record and comparison use.
"""


# def make_angle_list_v0(ind1,ind2):
#     """
#     Generate list of indices that match all angles for connections defined by (ind1,ind2) for each
#     unique index in ind1, meaning for each center. ind1 should be sorted.
#     Vectorized but requires memory for connections Max_bonds_per_atom*Number_atoms. Uses masking
    
#     Args:
#         ind1 (numpy array): Indexlist of start index for a bond. This must be sorted. Shape (N,)
#         ind2 (numpy array): Indexlist of end index for a bond. Shape (N,)
    
#     Returns:
#         out (numpy array):  Indexlist containing an angle-index-set. Shape (M,3)
#                             Where the angle is defined by 0-1-2 as 1->0,1->2 or 1<(0,2) 
#     """
#     import numpy as np
#     #agnles between bonds per atom, Reduced shape only for (N,)
#     indm = max(np.amax(ind1),np.amax(ind2))+1
#     bondtab = np.full(list(ind1.shape[:-1])+[indm,indm],-1)
#     bondtab[ind1,ind2] = ind2
#     bondtab = np.flip(np.sort(bondtab,axis=-1),axis=-1)
#     mask = bondtab!=-1
#     bmax = np.amax(np.sum(mask,axis=-1))
#     mask = mask[...,:bmax]
#     bondtab = bondtab[...,:bmax]
#     b1 = np.tile(np.expand_dims(np.expand_dims(np.arange(0,indm),axis=-1),axis=-1),(1,bmax,bmax))
#     b2 = np.tile(np.expand_dims(bondtab,axis=-1),(1,1,bmax))
#     b3 = np.tile(np.expand_dims(bondtab,axis=-2),(1,bmax,1))
#     mask2 = np.tile(np.expand_dims(mask,axis=-1),(1,1,bmax))
#     mask3 = np.tile(np.expand_dims(mask,axis=-2),(1,bmax,1))
#     mask = np.logical_and(mask2,mask3)
#     mask[...,np.arange(0,bmax),np.arange(0,bmax)]= False
#     b1 = b1[mask]
#     b2 = b2[mask]
#     b3 = b3[mask]
#     bcouples = np.concatenate([np.expand_dims(b2,axis=1),np.expand_dims(b3,axis=1)],axis=-1)
#     bcouples_sorted = np.sort(bcouples,axis=-1)
#     u,its = np.unique(bcouples_sorted,axis=0,return_index =True)
#     b1_u = b1[its]
#     sort1 = np.argsort(b1_u)
#     out1 = b1_u[sort1]
#     out2 = u[sort1]
#     return out1,out2