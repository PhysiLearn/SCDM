# -*- coding: utf-8 -*-
from preprocess.utils_test import*
import numpy as np
import pandas as pd
import os
atom_types = 119
wyck_types = 28
n_max = 24
# your csv here
csv_file = '../SCDM/POSCAR/merged_file.csv'
# your poscar data here
directory_path = '../POSCAR'


element_list = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
    'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re',
    'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
    'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl',
    'Mc', 'Lv', 'Ts', 'Og'
]


G,L,XYZ,A,W=GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max)


G= np.array(G)
L= np.array(L)
XYZ= np.array(XYZ)
A= np.array(A)
W= np.array(W)


np.savez('output_data.npz', G=G, L=L, XYZ=XYZ, A=A, W=W)

data = np.load('output_data.npz')
G = data['G']
L = data['L']
XYZ = data['XYZ']
A = data['A']
W = data['W']



import numpy as np
G_expanded = np.zeros((G.shape[0], 24, 3))
G_expanded[:, :, 0] = np.repeat(G[:, np.newaxis], 24, axis=1)  
G_expanded=G_expanded[:, np.newaxis, :, :]



L_expanded = np.zeros((L.shape[0], 24, 3))
L_expanded = np.repeat(np.expand_dims(L, axis=2), 4, axis=1) 
L_expanded = np.repeat(L_expanded, 3, axis=2)
L_expanded=L_expanded[:, np.newaxis, :, :]




def replace_elements_per_row(A):
    replacement_values = [ 1, 2, 5,7,9,12,15,18]  
    result = np.copy(A)
    
    for i in range(A.shape[0]):
        unique_elements = {}
        current_replacement_index = 0
        for j in range(A.shape[1]):
            element = A[i, j]
            if element == 0:
                continue  
            if element not in unique_elements:
                if current_replacement_index < len(replacement_values):
                    unique_elements[element] = replacement_values[current_replacement_index]
                    current_replacement_index += 1
            result[i, j] = unique_elements[element]
    
    return result


A_chuli = replace_elements_per_row(A)

A_expanded = np.repeat(np.expand_dims(A_chuli, axis=2), 3, axis=2)  
A_expanded=A_expanded[:, np.newaxis, :, :]


Wshape = W.shape
for i in range(Wshape[0]):
    for j in range(Wshape[1]):
        if W[i, j] != 0:
            W[i, j] = round(W[i, j] + 1, 2)
        else:
            W[i, j] = round(W[i, j] - 1, 2)

W_expanded = np.repeat(np.expand_dims(W, axis=2), 3, axis=2)  
W_expanded=W_expanded[:, np.newaxis, :, :]


xyzshape = XYZ.shape
for i in range(xyzshape[0]):
    for j in range(xyzshape[1]):
        for k in range(xyzshape[2]):
            XYZ[i, j, k] = round(XYZ[i, j, k], 2)

XYZ_expanded = XYZ[:, np.newaxis, :, :]

arrays_to_concat = (A_expanded,L_expanded,XYZ_expanded)

final_array = np.concatenate(arrays_to_concat, axis=1)
x = final_array

np.savez_compressed('output_data_wycoff.npz', x=x)



