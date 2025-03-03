import os
import numpy as np
from rdkit import Chem
import torch

np.random.seed(1)
EPSILON = 1e-08

def read_sdf_and_get_coordinates(sdf_file_path):
    suppl = Chem.SDMolSupplier(sdf_file_path)
    coordinates_list = []
    for mol in suppl:
        if mol is not None:
            conf = mol.GetConformer()
            coordinates = conf.GetPositions()
            coordinates_list.append(coordinates)
    coordinates_array = np.array(coordinates_list)
    return coordinates_array

def get_vec(crd):
    crd_ = np.roll(crd, -1, axis=-2)
    vec = crd_ - crd
    return vec[:, :-1, :]

def get_dis(crd):
    vec = get_vec(crd)
    dis = np.linalg.norm(vec, axis=-1, keepdims=True)
    return dis, vec

def get_angle(crd):
    dis, vec = get_dis(crd)
    vec_ = np.roll(vec, -1, axis=-2)
    dis_ = np.roll(dis, -1, axis=-2)
    angle = np.einsum('ijk,ijk->ij', vec, vec_)[..., None] / (dis * dis_ + EPSILON)
    return np.arccos(angle[:, :-1, :]), dis, vec

def get_dihedral(crd):
    angle, dis, vec_0 = get_angle(crd)
    vec_1 = np.roll(vec_0, -1, axis=-2)
    vec_2 = np.roll(vec_1, -1, axis=-2)
    vec_01 = np.cross(vec_0, vec_1)
    vec_12 = np.cross(vec_1, vec_2)
    vec_01 /= np.linalg.norm(vec_01, axis=-1, keepdims=True) + EPSILON
    vec_12 /= np.linalg.norm(vec_12, axis=-1, keepdims=True) + EPSILON
    dihedral = np.einsum('ijk,ijk->ij', vec_01, vec_12)[..., None]
    return np.arccos(dihedral[:, :-2, :]), angle, dis

def get_inner_crd(crd):
    dihedral, angle, dis = get_dihedral(crd)
    dihedral_ = np.pad(dihedral, ((0, 0), (3, 0), (0, 0)), mode='constant', constant_values=0)
    angle_ = np.pad(angle, ((0, 0), (2, 0), (0, 0)), mode='constant', constant_values=0)
    dis_ = np.pad(dis, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
    inner_crd = np.concatenate((dis_, angle_, dihedral_), axis=-1)
    return inner_crd

def get_element_numbers(sdf_file_path):
    suppl = Chem.SDMolSupplier(sdf_file_path)
    element_numbers_list = []
    for mol in suppl:
        if mol is not None:
            atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            element_numbers_list.append(atomic_nums)
    element_numbers_array = np.array(element_numbers_list)
    return element_numbers_array


data_folder = 'data'
file_names = os.listdir(data_folder)
all_icrd1, all_icrd2, all_icrd3, all_element_numbers = [], [], [], []

for file_name in file_names:
    file_path = os.path.join(data_folder, file_name)
    if file_name.endswith('.sdf'):
        origin_crd = read_sdf_and_get_coordinates(file_path)
        icrd = get_inner_crd(origin_crd)
        icrd = np.pad(icrd, ((0, 0), (0, 128 - icrd.shape[1]), (0, 0)), mode='constant', constant_values=0)
        element_numbers = get_element_numbers(file_path)
        element_numbers = np.pad(element_numbers, ((0, 0), (0, 128 - element_numbers.shape[1])), mode='constant',
                                 constant_values=0)
        all_icrd1.append(icrd[:, :, 0:1])
        all_icrd2.append(icrd[:, :, 1:2])
        all_icrd3.append(icrd[:, :, 2:3])
        # Expand dimension of element_numbers
        all_element_numbers.append(np.expand_dims(element_numbers, axis=-1))  # 添加一个新的维度

# Convert lists to numpy arrays
all_icrd1 = np.array(all_icrd1)
all_icrd2 = np.array(all_icrd2)
all_icrd3 = np.array(all_icrd3)
all_element_numbers = np.array(all_element_numbers)

# Concatenate arrays along the second axis to get the desired shape (144, 4, 128, 1)
all_data = np.concatenate((all_icrd1, all_icrd2, all_icrd3, all_element_numbers), axis=1)
#all_data = np.concatenate((all_icrd1, all_icrd2, all_icrd3), axis=1)
# Save the npz file
np.savez_compressed('mol_information.npz', x=all_data)