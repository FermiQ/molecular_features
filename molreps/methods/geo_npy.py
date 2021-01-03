"""
Molecular geometric feature representation based on numpy. A loose collection of functions.

Modular functions to compute distance matrix, angles, coordinates, connectivity, etc.
Many functions are written for batches too. Ideally all functions are vectorized.
Note: All functions are supposed to work out of the box without any dependencies, i.e. do not depend on each other.

@author: Patrick Reiser,
"""

import numpy as np


    
def coordinates_to_distancematrix(Coord3D):
    """
    Transform coordinates to distance matrix. Will apply transformation on last dimension.
    Changing of shape (...,N,3) -> (...,N,N)
    
    Arg:
        Coord3D (numpy array):  Coordinates of shape (...,N,3) for cartesian coordinates (x,y,z) 
                                and N the number of atoms or points. Coordinates are last dimension.

    Returns:
        Distance matrix as numpy array with shape (...,N,N) where N is the number of atoms
    """
    shape_3d = len(Coord3D.shape)
    a = np.expand_dims(Coord3D, axis = shape_3d -2)
    b = np.expand_dims(Coord3D, axis = shape_3d -1)
    c = b-a
    d = np.sqrt(np.sum(np.square(c),axis = shape_3d))
    return d


def invert_distance(d,nan=0,posinf=0,neginf=0):
    """
    Invert distance array, e.g. distance matrix. Inversion is done for all entries.
    Keeping of shape (...,) -> (...,)
    
    Args:
        d (numpy array): array of distance values of shape (...,)
        nan (value): replacement for np.nan after division, default = 0
        posinf (value): replacement for np.inf after division, default = 0
        neginf (value): replacement for -np.inf after division, default = 0
        
    Returns:
        Inverted distance array as numpy array of identical shape (...,) and
        replaces np.nan and np.inf with e.g. 0
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(1,d)
        #c[c == np.inf] = 0
        c = np.nan_to_num(c,nan=nan,posinf=posinf,neginf=neginf)
    return c


def inversedistancematrix_to_coulombmatrix(dinv,ProtonNumber):
    """
    Calculate Coulombmatrix from inverse distance Matrix plus nuclear charges/proton number.
    Transform shape as (...,N,N) + (...,N) -> (...,N,N)
    
    Args:
        dinv (numpy array): Inverse distance matrix defined at last two axis.
                            Array of shape (...,N,N) with N number of atoms storing inverse distances.
        ProtonNumber (numpy array): Nuclear charges given in last dimension. 
                                    Order must match entries in inverse distance matrix.
                                    array of shape (...,N)
    
    Returns:
        Numpy array with Coulombmatrix at last two dimension (...,N,N). 
        Function multiplies Z_i*Z_j with 1/d_ij and set diagonal to 0.5*Z_ii^2.4
    """  
    shape_z = ProtonNumber.shape
    a = np.expand_dims(ProtonNumber, axis = len(shape_z)-1)
    b = np.expand_dims(ProtonNumber, axis = len(shape_z))
    c = a*b
    coul = dinv*c
    indslie = np.arange(0,shape_z[-1])
    coul[...,indslie,indslie] = np.power(ProtonNumber, 2.4)*0.5
    return coul


def value_to_oneHot(vals,compare):
    """
    Convert array of values e.g. nuclear charge to one-hot representation thereof. 
    A dictionary of all possible values is required. 
    Expands shape from (...,) + (M,) -> (...,M)
    
    Args:
        vals (numpy array): array of values to convert.
        compare (numpy array): 1D-numpy array with a list of possible values.
        
    Returns:
        A one-hot representation of vals input with expanded last dimension to match
        the compare dictionary. Entries are 1.0 if vals == compare[i] and 0.0 else
    """
    comp = np.array(compare,dtype=vals.dtype)
    vals_shape = vals.shape
    vals = np.expand_dims(vals,axis=-1)
    comp = np.broadcast_to(comp,vals_shape+comp.shape) #shape (1,1,...,M)
    out = np.array(vals == comp,dtype=np.float32)
    return out


def coulombmatrix_to_inversedistance_proton(self,coulmat,unit_conversion=1):
    """ 
    Converts a coulomatrix back to inverse distancematrix + atomic number.
    (...,N,N)-> (...,N,N)+(...,N)
    
    Args:
        Coulombmatrix (numpy array): Full Coulombatrix of shape (...,N,N)
        unit_conversion (float) : Whether to scale units for distance. Default is 1.
    
    Returns: 
        Inverse Distance Matrix (numpy array) : Inverse distance of (...,N,N)
        Atom Number (numpy array) : Corresponding diagonal as proton number
    """
    indslie = np.arange(0,coulmat.shape[-1])
    z = coulmat[...,indslie,indslie]
    z = np.power(2*z,1/2.4)
    a = np.expand_dims(z, axis = len(z.shape)-1)
    b = np.expand_dims(z, axis = len(z.shape))
    zz = a*b
    c = coulmat/zz
    c[...,indslie,indslie]  = 0
    c /= unit_conversion
    z = np.array(np.round(z),dtype=np.int)
    return c,z


def distance_to_gaussdistance(Distance, GBins = 30, GRange = 5.0, GSigma = 0.2):
    """ 
    Convert distance array to smooth one-hot representation using Gaussian functions.
    Changes shape for gaussian distance (...,) -> (...,GBins)
    The Default values match units in Angstroem.
    
    Args:
        Distance (numpy array): Array of distances of shape (...,)
        GBins (int): number of Bins to sample distance from, default = 30
        GRange (value): maximum distance to be captured by bins, default = 5.0
        GSigma (value): sigma of the gaussian function, determining the width/sharpness, default = 0.2
    
    Returns:
        Numpy array of gaussian distance with expanded last axis (...,GBins)
    """
    gamma = 1/GSigma/GSigma*(-1)/2
    d_shape = Distance.shape
    EdgeDistGrid= np.expand_dims(Distance,axis=-1)
    EdgeGausBin = np.arange(0,GBins,1)/GBins*GRange
    EdgeGausBin = np.broadcast_to(EdgeGausBin, np.append(np.ones(len(d_shape),dtype=np.int32),EdgeGausBin.shape)) #shape (1,1,...,GBins)
    EdgeGausBin = np.square(EdgeDistGrid-EdgeGausBin)*gamma # (N,M,...,1) - (1,1,...,GBins)
    EdgeGausBin = np.exp(EdgeGausBin)
    return EdgeGausBin


def sort_distmatrix(DistanceMatrix):
    """
    Sort a flexible shaped distance matrix along last dimension
    Keeps shape (...,N,M) -> index (...,N,M) + sorted (...,N,M)
    
    Args:
        DistanceMatrix (numpy array): Matrix of distances of shape (...,N,M)
    
    Returns:
        SortingIndex (numpy array): Indices of sorted last dimension entries. Shape (...,N,M)
        SortedDistance (numpy array): Sorted Distance Matrix, sorted at last dimension.
    """
    SortingIndex = np.argsort(DistanceMatrix,axis=-1)
    SortedDistance = np.take_along_axis(DistanceMatrix, SortingIndex, axis=-1)
    return SortingIndex,SortedDistance 



def get_connectivity_from_inversedistancematrix(invdistmat,protons,radii_dict=None, k1=16.0, k2=4.0/3.0, cutoff=0.85,force_bonds=True): 
    """
    Get connectivity table from inverse distance matrix defined at last dimensions (...,N,N) and corresponding bond-radii. Keeps shape with (...,N,N).
    Covalent radii, from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197. 
    Values for metals decreased by 10% according to Robert Paton's Sterimol implementation. 
    Partially based on code from Robert Paton's Sterimol script, which based this part on Grimme's D3 code
    
    Args:
        invdistmat (numpy array):   inverse distance matrix defined at last dimensions (...,N,N)
                                    distances must be in Angstroem not in Bohr  
        protons (numpy array):      An array of atomic numbers matching the invdistmat (...,N),
                                    for which the radii are to be computed.
        radii_dict (numpy array):   covalent radii for each element. If default=None, stored values are used. Otherwise array with covalent bonding radii.
                                    example: np.array([0, 0.24, 0.46, 1.2, ...]) from {'H': 0.34, 'He': 0.46, 'Li': 1.2, ...}
        k1 (value):                 default = 16
        k2 (value):                 default = 4.0/3.0
        cutoff (value):             cutoff value to set values to Zero (no bond) default = 0.85
        force_bonds (value):        whether to force at least one bond in the bond table per atom (default = True)
        
    Retruns:
        Connectivity table with 1 for chemical bond and zero otherwise of shape (...,N,N) -> (...,N,N)
    """
    #Dictionary of bond radii
    original_radii_dict = {'H': 0.34, 'He': 0.46, 'Li': 1.2, 'Be': 0.94, 'B': 0.77, 'C': 0.75, 'N': 0.71, 'O': 0.63, 'F': 0.64, 'Ne': 0.67, 'Na': 1.4, 'Mg': 1.25, 'Al': 1.13, 'Si': 1.04, 'P': 1.1, 'S': 1.02, 'Cl': 0.99, 'Ar': 0.96, 'K': 1.76, 'Ca': 1.54, 'Sc': 1.33, 'Ti': 1.22, 'V': 1.21, 'Cr': 1.1, 'Mn': 1.07, 'Fe': 1.04, 'Co': 1.0, 'Ni': 0.99, 'Cu': 1.01, 'Zn': 1.09, 'Ga': 1.12, 'Ge': 1.09, 'As': 1.15, 'Se': 1.1, 'Br': 1.14, 'Kr': 1.17, 'Rb': 1.89, 'Sr': 1.67, 'Y': 1.47, 'Zr': 1.39, 'Nb': 1.32, 'Mo': 1.24, 'Tc': 1.15, 'Ru': 1.13, 'Rh': 1.13, 'Pd': 1.19, 'Ag': 1.15, 'Cd': 1.23, 'In': 1.28, 'Sn': 1.26, 'Sb': 1.26, 'Te': 1.23, 'I': 1.32, 'Xe': 1.31, 'Cs': 2.09, 'Ba': 1.76, 'La': 1.62, 'Ce': 1.47, 'Pr': 1.58, 'Nd': 1.57, 'Pm': 1.56, 'Sm': 1.55, 'Eu': 1.51, 'Gd': 1.52, 'Tb': 1.51, 'Dy': 1.5, 'Ho': 1.49, 'Er': 1.49, 'Tm': 1.48, 'Yb': 1.53, 'Lu': 1.46, 'Hf': 1.37, 'Ta': 1.31, 'W': 1.23, 'Re': 1.18, 'Os': 1.16, 'Ir': 1.11, 'Pt': 1.12, 'Au': 1.13, 'Hg': 1.32, 'Tl': 1.3, 'Pb': 1.3, 'Bi': 1.36, 'Po': 1.31, 'At': 1.38, 'Rn': 1.42, 'Fr': 2.01, 'Ra': 1.81, 'Ac': 1.67, 'Th': 1.58, 'Pa': 1.52, 'U': 1.53, 'Np': 1.54, 'Pu': 1.55}
    proton_raddi_dict = np.array([0, 0.34, 0.46, 1.2, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.4, 1.25, 1.13, 1.04, 1.1, 1.02, 0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.1, 1.07, 1.04, 1.0, 0.99, 1.01, 1.09, 1.12, 1.09, 1.15, 1.1, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.19, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51, 1.52, 1.51, 1.5, 1.49, 1.49, 1.48, 1.53, 1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32, 1.3, 1.3, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58, 1.52, 1.53, 1.54, 1.55])
    if(radii_dict== None):
        radii_dict = proton_raddi_dict #index matches atom number
    #Get Radii 
    protons = np.array(protons,dtype=np.int)
    radii = radii_dict[protons]
    # Calculate
    shape_rad = radii.shape
    r1 = np.expand_dims(radii, axis = len(shape_rad)-1)
    r2 = np.expand_dims(radii, axis = len(shape_rad))
    rmat = r1+r2
    rmat = k2*rmat
    rr = rmat*invdistmat
    damp = (1.0+np.exp(-k1*(rr-1.0)))
    damp = 1.0/damp        
    if(force_bonds==True): # Have at least one bond
        print("Info: Forcing bond.")
        maxvals = np.expand_dims(np.argmax(damp,axis=-1),axis=-1)
        np.put_along_axis(damp,maxvals,1,axis=-1)
        #To make it symmetric transpose last two axis
        damp = np.swapaxes(damp,-2,-1)
        np.put_along_axis(damp,maxvals,1,axis=-1)
        damp = np.swapaxes(damp,-2,-1)
    damp[damp<cutoff] = 0
    bond_tab = np.round(damp)          
    return bond_tab


def get_indexmatrix(shape,flatten=False):
    """
    Matrix of indices with a_ijk... = [i,j,k,..] for shape (N,M,...,len(shape)) with Indexlist being the last dimension.
    Note: numpy indexing does not work this way but as indexlist per dimension
    
    Args:
        shape (list, int): list of target shape, e.g. (2,2)
        flatten (bool): whether to flatten the output or keep inputshape, default=False
    
    Returns: 
        Index array of shape (N,M,...,len(shape)) e.g. [[[0,0],[0,1]],[[1,0],[1,1]]]
    """
    indarr = np.indices(shape)
    re_order =np.append(np.arange(1,len(shape)+1),0)
    indarr = indarr.transpose(re_order)
    if(flatten==True):
        indarr = np.reshape(indarr,(np.prod(shape),len(shape)))
    return indarr


def coordinates_from_distancematrix(DistIn,use_center = None,dim=3):
    """
    Computes list of coordinates from a distance matrix of shape (N,N)
    
    Args:
        DistIn (numpy array): Distance matrix of shape (N,N) with D_ij = |r_i-r_j|
        use_center (int): which atom should be the center, dafault = None means center of mass
        dim (int): the dimension of embedding, 3 is default
    
    Return:
        List of Atom coordinates [[x_1,x_2,x_3],[x_1,x_2,x_3],...] 
    
    Uses vectorized Alogrithm:
        http://scripts.iucr.org/cgi-bin/paper?S0567739478000522
        https://www.researchgate.net/publication/252396528_Stable_calculation_of_coordinates_from_distance_information
    no check of positive semi-definite or possible k-dim >= 3 is done here
    performs svd from numpy
    may even wok for (...,N,N) but not tested
    """
    DistIn = np.array(DistIn)
    dimIn = DistIn.shape[-1]   
    if use_center is None:
        #Take Center of mass (slightly changed for vectorization assuming d_ii = 0)
        di2 = np.square(DistIn)
        di02 = 1/2/dimIn/dimIn*(2*dimIn*np.sum(di2,axis=-1)-np.sum(np.sum(di2,axis=-1),axis=-1))
        MatM = (np.expand_dims(di02,axis=-2) + np.expand_dims(di02,axis=-1) - di2)/2 #broadcasting
    else:
        di2 = np.square(DistIn)
        MatM = (np.expand_dims(di2[...,use_center],axis=-2) + np.expand_dims(di2[...,use_center],axis=-1) - di2 )/2
    u,s,v = np.linalg.svd(MatM)
    vecs = np.matmul(u,np.sqrt(np.diag(s))) # EV are sorted by default
    distout = vecs[...,0:dim]
    return distout


def make_rotationmatrix(vector,angle):
    """
    Generate rotationmatrix around a given vector with a certain angle.
    Only defined for 3 dimensions here.
    
    Args:
        vector (numpy array, list): vector of rotation axis (3,) with (x,y,z)
        angle (value): angle in degrees ° to rotate around
    
    Returns:
        rotation matrix R of shape (3,3) that performs the rotation with y = R*x
    """
    angle=angle/180.0*np.pi
    norm=(vector[0]**2.0+vector[1]**2.0+vector[2]**2.0)**0.5
    direction=vector/norm    
    matrix=np.zeros((3,3))
    matrix[0][0]=direction[0]**2.0*(1.0-np.cos(angle))+np.cos(angle)
    matrix[1][1]=direction[1]**2.0*(1.0-np.cos(angle))+np.cos(angle)
    matrix[2][2]=direction[2]**2.0*(1.0-np.cos(angle))+np.cos(angle)    
    matrix[0][1]=direction[0]*direction[1]*(1.0-np.cos(angle))-direction[2]*np.sin(angle)
    matrix[1][0]=direction[0]*direction[1]*(1.0-np.cos(angle))+direction[2]*np.sin(angle)    
    matrix[0][2]=direction[0]*direction[2]*(1.0-np.cos(angle))+direction[1]*np.sin(angle)
    matrix[2][0]=direction[0]*direction[2]*(1.0-np.cos(angle))-direction[1]*np.sin(angle)    
    matrix[1][2]=direction[1]*direction[2]*(1.0-np.cos(angle))-direction[0]*np.sin(angle)
    matrix[2][1]=direction[1]*direction[2]*(1.0-np.cos(angle))+direction[0]*np.sin(angle)    
    return(matrix)
    

def rotate_to_principle_axis(coord):
    """ 
    Rotate a pointcloud to its principle axis. This can be a molecule but also some general data.
    It uses PCA via SVD from numpy.linalg.svd(). PCA from scikit uses SVD too (scipy.sparse.linalg).
    
    Args:
        coord (numpy array): Array of points forming a pointcloud. Important: coord has shape (N,p)
                             where N is the number of samples and p is the feature/coordinate dimension e.g. 3 for x,y,z
    
    Returns:
        R (numpy array): rotaton matrix of shape (p,p) if input has (N,p)
        rotated (numpy array): rotated pointcould of coord that was the input.
    
    Note:
        The data is centered before SVD but shifted back at the output.
    """
    centroid_c = np.mean(coord,axis=0)
    sm = coord - centroid_c
    zzt = (np.dot(sm.T,sm)) # Calculate covariance matrix
    u, s, vh  = np.linalg.svd(zzt)
    #Alternatively SVD of coord with onyl compute vh but not possible for numpy/scipy
    rotated = np.dot(sm,vh.T)
    rotshift = rotated + centroid_c
    return vh, rotshift


def rigid_transform(A, B,correct_reflection=False):
    """ 
    Rotate and shift pointcloud A to pointcloud B. This should implement Kabsch algorithm.
    
    Important: the numbering of points of A and B must match, no shuffled pointcloud.
    This works for 3 dimensions only. Uses SVD.
    
    Args:
        A (numpy array): list of points (N,3) to rotate (and translate)
        B (numpy array): list of points (N,3) to rotate towards: A to B, where the coordinates (3) are (x,y,z)
    
    Returns:
        A_rot (numpy array): Rotated and shifted version of A to match B
        R (numpy array): Rotation matrix
        t (numpy array): translation from A to B
        
    Note: 
        Explanation of Kabsch Algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm
        For further literature
        https://link.springer.com/article/10.1007/s10015-016-0265-x
        https://link.springer.com/article/10.1007%2Fs001380050048
        maybe work for (...,N,3), not tested
    """
    A = np.transpose(np.array(A))
    B = np.transpose(np.array(B))
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    Am = A - np.expand_dims(centroid_A,axis=1)
    Bm = B - np.expand_dims(centroid_B,axis=1)
    H = np.dot(Am ,np.transpose(Bm))
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T,U.T)
    d = np.linalg.det(R)
    if(d<0):
        print("Warning: det(R)<0, det(R)=",d)
        if(correct_reflection==True):
            print("Correcting R...")
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
    Bout = np.dot(R,Am) + np.expand_dims(centroid_B,axis=1)
    Bout = np.transpose(Bout)
    t = np.expand_dims(centroid_B-np.dot(R,centroid_A),axis=0)
    t = t.T
    return Bout,R, t


def get_angles(coords,inds):
    """
    Compute angeles between coordinates (...,N,3) from a matching index list that has shape (...,M,3) with (ind0,ind1,ind2).
    Angles are between ind1<(ind0,ind2) taking coords[ind]. The angle is oriented as ind1->ind0,ind1->ind2.
    
    Args:
        coords (numpy array): list of coordinates of points (...,N,3)    
        inds (numpy array): Index list of points (...,M,3) that means coords[i] with i in axis=-1.
    
    Returns:
        angle_sin (numpy array): sin() of the angles between ind2<(ind1,ind3)
        angle_cos (numpy array): cos() of the angles between ind2<(ind1,ind3)
        angles (numpy array): angles in rads
        norm_vec1 (numpy array): length of vector ind1,ind2a
        norm_vec2 (numpy array): length of vector ind1,ind2b
    """
    ind1 = inds[...,1]
    ind2a = inds[...,0]
    ind2b = inds[...,2]
    vcords1 = np.take_along_axis(coords,np.expand_dims(ind1,axis=-1),axis=-2)
    vcords2a = np.take_along_axis(coords,np.expand_dims(ind2a,axis=-1),axis=-2)
    vcords2b = np.take_along_axis(coords,np.expand_dims(ind2b,axis=-1),axis=-2)
    vec1=-vcords1+vcords2a
    vec2=-vcords1+vcords2b
    norm_vec1 = np.sqrt(np.sum(vec1*vec1,axis=-1))
    norm_vec2 = np.sqrt(np.sum(vec2*vec2,axis=-1))
    angle_cos = np.sum(vec1*vec2,axis=-1)/ norm_vec1 /norm_vec2
    angles  =  np.arccos(angle_cos)
    angle_sin = np.sin(angles)
    return angle_sin,angle_cos,angles ,norm_vec1,norm_vec2


def all_angle_combinations(ind1,ind2):
    """
    Get all angles between ALL possible bonds also unrelated bonds e.g. (1,2) and (17,20) which are not connected. Input shape is (...,N).
    Note: This is mostly unpractical and not wanted, see make_angle_list for normal use.
    
    Args:
        ind1 (numpy array): Indexlist of start index for a bond. This must be sorted. Shape (...,N)
        ind2 (numpy array): Indexlist of end index for a bond. Shape (...,N)
    
    Returns
        index touples of shape (...,N*N/2-N,2,2) where the bonds are specified at last axis and
        the bond pairs at axis=-2
    """
    #For all angels between unconncected bonds, possible for (...,N)
    indb = np.concatenate([np.expand_dims(ind1,axis=-1),np.expand_dims(ind2,axis=-1)],axis=-1)
    tils = [1]*(len(indb.shape)+1)
    tils[-2] = ind1.shape[-1]
    b1 = np.tile(np.expand_dims(indb,axis=-2),tils)
    tils = [1]*(len(indb.shape)+1)
    tils[-3] = ind1.shape[-1]
    b2 = np.tile(np.expand_dims(indb,axis=-3),tils)
    bcouples = np.concatenate([np.expand_dims(b1,axis=-2),np.expand_dims(b2,axis=-2)],axis=-2)
    tris = np.tril_indices(ind1.shape[-1],k=-1)
    bcouples = bcouples[...,tris[0],tris[1],:,:]
    return bcouples


def make_angle_list(ind1,ind2):
    """
    Generate list of indices that match all angles for connections defined by (ind1,ind2) for each
    unique index in ind1, meaning for each center. ind1 should be sorted.
    Vectorized but requires memory for connections Max_bonds_per_atom*Number_atoms. Uses masking
    
    Args:
        ind1 (numpy array): Indexlist of start index for a bond. This must be sorted. Shape (N,)
        ind2 (numpy array): Indexlist of end index for a bond. Shape (N,)
    
    Returns:
            out (numpy array):  Indexlist containing an angle-index-set. Shape (M,3)
                                Where the angle is defined by 0-1-2 as 1->0,1->2 or 1<(0,2) 
    """
    #Get unique atoms as center for bonds
    n1_uni, n1_counts = np.unique(ind1,return_counts=True)
    n1_multi = np.repeat(n1_uni,n1_counts)
    max_bonds = np.max(n1_counts)
    # Make a list with (N_atoms,max_bonds) with zero padding for less bonds plus mask
    # this is btab, btab_values where values have index2
    btab = np.tile(np.expand_dims(np.arange(np.max(max_bonds)),axis=0),(len(n1_uni),1))
    btab = btab < np.expand_dims(n1_counts,axis=1)
    btab_flat = btab.flatten()
    btab_ind_flat = np.arange(len(btab_flat))
    btab_ind_flat_activ = btab_ind_flat[btab_flat]
    btab_values_flat = np.zeros(len(btab_flat))
    btab_values_flat[btab_ind_flat_activ] = ind2
    btab_values = np.reshape(btab_values_flat,btab.shape)
    # Expand this padded list to a matrix (N_atoms,max_bonds,max_bonds,2) to have all combinations like distance matrix and last dim indices
    # Mask of this matrix must have left/upper blocks and a diagonal set to zero (no angle between same atom)
    btab_values1 = np.tile(np.expand_dims(btab_values,axis=1),(1,max_bonds,1))
    btab_values2 = np.tile(np.expand_dims(btab_values,axis=2),(1,1,max_bonds))
    btab1 = np.tile(np.expand_dims(btab,axis=1),(1,max_bonds,1))
    btab2 = np.tile(np.expand_dims(btab,axis=2),(1,1,max_bonds))
    btab_values = np.concatenate([np.expand_dims(btab_values1,axis=-1),np.expand_dims(btab_values2,axis=-1)],axis=-1)
    btab_mat = np.logical_and(btab1,btab2)
    btab_mat[...,np.arange(0,max_bonds),np.arange(0,max_bonds)] = False #set diagonal to zero
    #Make the same matrix for the centers i.e. (N_atoms,max_bonds,max_bonds) with (...,max_bonds,max_bonds) has index of axis=0
    center_1 = np.tile(np.expand_dims(np.expand_dims(np.arange(len(btab_mat)),axis=-1),axis=-1) ,(1,max_bonds,max_bonds))
    # Take Mask to get a list of index couples.
    # The indices of bonds from center must be sorted to remove duplicates e.g. 0,2 and 2,0 will be same anlge 
    center_1 = center_1[btab_mat]
    bcouples = btab_values[btab_mat]
    bcouples_sorted = np.sort(bcouples,axis=-1)
    out_ind = np.concatenate([np.expand_dims(bcouples_sorted[:,0],axis=-1),
                              np.expand_dims(center_1,axis=-1),
                              np.expand_dims(bcouples_sorted[:,1],axis=-1),
                              ],axis=-1)
    #remove duplicate 'angles'
    out = np.unique(out_ind,axis=0)
    return out


def define_adjacency_from_distance(DistanceMatrix,max_distance=np.inf,max_neighbours=np.inf,exclusive=True,self_loops=False):
    """
    Construct adjacency matrix from a distance matrix by distance and number of neighbours. Works for batches.
    
    This does take into account special bonds (e.g. chemical) just a general distance measure.
    Tries to connect nearest neighbours.

    Args:
        DistanceMatrix (np.array): Distance Matrix of shape (...,N,N)
        max_distance (float, optional): Maximum distance to allow connections, can also be None. Defaults to np.inf.
        max_neighbours (int, optional): Maximum number of neighbours, can also be None. Defaults to np.inf.
        exclusive (bool, optional): Whether both max distance and Neighbours must be fullfileed. Defaults to True.
        self_loops (bool, optional): Allow self-loops on diagonal. Defaults to False.

    Returns:
        GraphAdjacency (np.array): Adjacency Matrix of shape (...,N,N) of dtype=np.bool.
        GraphIndices (np.array): Flatten indizes from former array that have Adjacency == True.

    """
    DistanceMatrix = np.array(DistanceMatrix)
    NumAtoms = DistanceMatrix.shape[-1]
    if(exclusive==True):
        GraphAdjacency = np.ones_like(DistanceMatrix,dtype = np.bool)
    else:
        GraphAdjacency = np.zeros_like(DistanceMatrix,dtype = np.bool)
    inddiag = np.arange(NumAtoms) 
    #Make Indix Matrix
    indarr = np.indices(DistanceMatrix.shape)
    re_order =np.append (np.arange(1,len(DistanceMatrix.shape)+1),0)
    GraphIndices = indarr.transpose(re_order) 
    #print(GraphIndices.shape)
    #Add Max Radius
    if(max_distance is not None):
        temp = DistanceMatrix < max_distance
        #temp[...,inddiag,inddiag] = False
        if(exclusive==True):
            GraphAdjacency = np.logical_and(GraphAdjacency, temp)
        else:
            GraphAdjacency = np.logical_or(GraphAdjacency, temp)
    # Add #Nieghbours
    if(max_neighbours is not None):
        max_neighbours = min(max_neighbours,NumAtoms)
        SortingIndex = np.argsort(DistanceMatrix,axis=-1)
        #SortedDistance = np.take_along_axis(self.DistanceMatrix, SortingIndex, axis=-1)
        ind_sorted_red = SortingIndex[...,:max_neighbours+1]
        temp = np.zeros_like(DistanceMatrix,dtype = np.bool)
        np.put_along_axis(temp,ind_sorted_red, True,axis=-1)
        if(exclusive==True):
            GraphAdjacency = np.logical_and(GraphAdjacency, temp)
        else:
            GraphAdjacency = np.logical_or(GraphAdjacency, temp)
    # Allow self-loops
    if(self_loops == False):
        GraphAdjacency[...,inddiag,inddiag] = False

    GraphIndices = GraphIndices[GraphAdjacency]
    return GraphAdjacency,GraphIndices

