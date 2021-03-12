from molreps.methods.geo_npy import coordinates_to_distancematrix,invert_distance,get_connectivity_from_inversedistancematrix
import matplotlib.pyplot as plt
import numpy as np

c1 = np.array([[0.000517, 0.000000 , 0.000299],
[0.000517 ,0.000000 , 1.394692],
[1.208097, 0.000000 , 2.091889],
[2.415677, 0.000000 , 1.394692],
[2.415677, 0.000000 , 0.000299],
[1.208097, 0.000000 ,-0.696898],
[-0.939430 ,0.000000, -0.542380],
[-0.939430 ,0.000000,  1.937371],
[1.208097 ,0.000000 , 3.177246],
[3.355625, 0.000000 , 1.937371],
[3.355625, 0.000000 ,-0.542380],
[1.208097, 0.000000 ,-1.782255 ]])
a1 = ['C','C','C','C','C','C','H','H','H','H','H','H']

d = coordinates_to_distancematrix(c1)
invd = invert_distance(d)

GlobalProtonDict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'b': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118, 'Uue': 119}
a2 = np.array([GlobalProtonDict[x] for x in a1])
bondtab = get_connectivity_from_inversedistancematrix(invd,a2)


def get_bonds(coords, elements, force_bonds=False, forced_bonds=[],cutoff=0.85): # covalent radii, from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197
    # values for metals decreased by 10% according to Robert Paton's Sterimol implementation
    rcov = {
    "H": 0.34,"He": 0.46,"Li": 1.2,"Be": 0.94,"b": 0.77,"C": 0.75,"N": 0.71,"O": 0.63,"F": 0.64,"Ne": 0.67,"Na": 1.4,"Mg": 1.25,"Al": 1.13,"Si": 1.04,"P": 1.1,"S": 1.02,"Cl": 0.99,"Ar": 0.96,"K": 1.76,"Ca": 1.54,"Sc": 1.33,"Ti": 1.22,"V": 1.21,"Cr": 1.1,"Mn": 1.07,"Fe": 1.04,"Co": 1.0,"Ni": 0.99,"Cu": 1.01,"Zn": 1.09,"Ga": 1.12,"Ge": 1.09,"As": 1.15,"Se": 1.1,"Br": 1.14,"Kr": 1.17,"Rb": 1.89,"Sr": 1.67,"Y": 1.47,"Zr": 1.39,"Nb": 1.32,"Mo": 1.24,"Tc": 1.15,"Ru": 1.13,"Rh": 1.13,"Pd": 1.19,"Ag": 1.15,"Cd": 1.23,"In": 1.28,"Sn": 1.26,"Sb": 1.26,"Te": 1.23,"I": 1.32,"Xe": 1.31,"Cs": 2.09,"Ba": 1.76,"La": 1.62,"Ce": 1.47,"Pr": 1.58,"Nd": 1.57,"Pm": 1.56,"Sm": 1.55,"Eu": 1.51,"Gd": 1.52,"Tb": 1.51,"Dy": 1.5,"Ho": 1.49,"Er": 1.49,"Tm": 1.48,"Yb": 1.53,"Lu": 1.46,"Hf": 1.37,"Ta": 1.31,"W": 1.23,"Re": 1.18,"Os": 1.16,"Ir": 1.11,"Pt": 1.12,"Au": 1.13,"Hg": 1.32,"Tl": 1.3,"Pb": 1.3,"Bi": 1.36,"Po": 1.31,"At": 1.38,"Rn": 1.42,"Fr": 2.01,"Ra": 1.81,"Ac": 1.67,"Th": 1.58,"Pa": 1.52,"U": 1.53,"Np": 1.54,"Pu": 1.55
    } # partially based on code from Robert Paton's Sterimol script, which based this part on Grimme's D3 code
    natom = len(coords)
    #max_elem = 94
    k1 = 16.0
    k2 = 4.0/3.0
    conmat = np.zeros((natom,natom))
    bonds = []
    for i in range(0,natom):
        if elements[i] not in rcov.keys():
            continue
        for iat in range(0,natom):
            if elements[iat] not in rcov.keys():
                continue
            if iat != i:
                dx = coords[iat][0] - coords[i][0]
                dy = coords[iat][1] - coords[i][1]
                dz = coords[iat][2] - coords[i][2]
                r = np.linalg.norm([dx,dy,dz])
                rco = rcov[elements[i]]+rcov[elements[iat]]
                rco = rco*k2
                rr=rco/r
                damp=1.0/(1.0+np.math.exp(-k1*(rr-1.0)))
                if damp > cutoff: #check if threshold is good enough for general purpose
                    conmat[i,iat],conmat[iat,i] = 1,1
                pair=[min(i,iat),max(i,iat)]
                if pair not in bonds: # add some empirical rules here:
                    is_bond=True
                    #elements_bond = [elements[pair[0]], elements[pair[1]]]
                if is_bond:
                    bonds.append(pair)
    return conmat

bondtab2 = get_bonds(c1,a1)

print(bondtab)
print(bondtab2)
print(bondtab-bondtab2)