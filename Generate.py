import os
# import sys
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from torch.utils.data import DataLoader,Dataset
from torch import optim
from torch.optim import Adam
from UNet import Unet
from simpleDiffusion.simpleDiffusion import DiffusionModel
from utils.trainNetworkHelper import SimpleDiffusionTrainer
#your npz data here：same with train
dataname='output_data_poscar9k_A125_L_XYZ10_2.npz'
data=np.load('output_data_poscar9k_A125_L_XYZ10_2.npz')
x=data['x']

x=torch.Tensor(x).to(torch.float)
print('x',type(x))

#your poscar csv here :same with train
csv_file = '../poscar-1/merged_file.csv'

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print('y',self.y)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        features = self.x[idx]
        label = self.y[idx]
        return features, label


image_size = 24
channels = 3
batch_size = 24


labels_df = pd.read_csv(csv_file)

labels=labels_df.id.values

custom_dataset = CustomDataset(x, labels)

data_loader = DataLoader(custom_dataset, batch_size=batch_size)
for features, labels in data_loader:
    print('Features shape in a batch:', features.shape)
    print('Labels shape in a batch:', labels.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"
dim_mults = (1, 2, 4)

denoise_model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=dim_mults
)


timesteps = 1000
schedule_name = "linear_beta_schedule"
DDPM = DiffusionModel(schedule_name=schedule_name,
                      timesteps=timesteps,
                      beta_start=0.0001,
                      beta_end=0.02,
                      denoise_model=denoise_model).to(device)


optimizer = Adam(DDPM.parameters(), lr=1e-3)
epoches = 1000

Trainer = SimpleDiffusionTrainer(epoches=epoches,
                                 train_loader=data_loader,
                                 optimizer=optimizer,
                                 device=device,
                                 timesteps=timesteps)


root_path = "./saved_train_models"
setting = "imageSize{}_channels{}_dimMults{}_epoches{}_dataname{}".format(image_size, channels, dim_mults, epoches, dataname)

saved_path = os.path.join(root_path, setting)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

best_model_path = saved_path + '/' + 'BestModel.pth'
#your trained model here
best_model_path ='../saved_train_models/BestModel.pth'
DDPM.load_state_dict(torch.load(best_model_path)) 

sample_data = []
num_samples = 150
batch_size = 24

num_batches = num_samples // batch_size
remainder = num_samples % batch_size

for _ in range(num_batches):
    samples = DDPM(mode="generate", image_size=24, batch_size=24, channels=channels)
    generated_images = samples[-1][:batch_size]
    for i in range(batch_size):
        generate_image = generated_images[i].reshape(channels, image_size, 3)
        sample_data.append(generate_image)

if remainder > 0:
    samples = DDPM(mode="generate", image_size=24, batch_size=24, channels=channels)
    generated_images = samples[-1][:remainder]
    for i in range(remainder):
        generate_image = generated_images[i].reshape(channels, image_size, 3)
        sample_data.append(generate_image)


print(sample_data)
    
#your sample data
save_path = "sample_data.npz"
np.savez_compressed(save_path, samples=np.array(sample_data))


import numpy as np
import pandas as pd


# load your sample data here 
data = np.load('../sample_data.npz', allow_pickle=True)
final_array = data['samples']

arrays = np.split(final_array, 3, axis=1)

num_batches=num_samples

A_decoded = arrays[0]
A_reduced = np.zeros((num_batches, 24), dtype=np.int32)

for i in range(A_decoded.shape[0]):
    for j in range(A_decoded.shape[2]):
        current_slice = A_decoded[i, 0, j, :]
        current_slice[(current_slice >= 0.5) & (current_slice <= 1.5)] = 1
        current_slice[(current_slice > 1.5) & (current_slice <= 3)] = 2
        current_slice[current_slice > 3] = 5
        A_reduced[i, j] = int(np.mean(current_slice))
A_processed = A_reduced


L_decoded = arrays[1]
L_processed = np.zeros((L_decoded.shape[0], 6))

for i in range(L_decoded.shape[0]):
    L = L_decoded[i, 0] 
    L_avg = np.mean(L, axis=1, keepdims=True)
    temp_output = np.zeros((6,))
    for j in range(0, L_avg.shape[0], 4):
        temp_output[j//4] = np.mean(L_avg[j:j+4])
    L_processed[i] = temp_output



def lattice_matrix(a, b, c, alpha, beta, gamma):
    """
    Calculate the lattice matrix from lattice constants and angles.
    
    Args:
    a, b, c (float): Lattice constants.
    alpha, beta, gamma (float): Angles in radians.
    
    Returns:
    np.ndarray: 3x3 lattice matrix.
    """
    # Calculate the lattice vectors
    vector_a = np.array([a, 0, 0])
    
    vector_b = np.array([
        b * np.cos(gamma),
        b * np.sin(gamma),
        0
    ])
    
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    
    vector_c = np.array([
        c * cos_beta,
        c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
        c * np.sqrt(1 - cos_beta**2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2)
    ])
    
    lattice_matrix = np.vstack((vector_a, vector_b, vector_c))
    return lattice_matrix

input_array = L_processed

L_prove = []

for params in input_array:
    a, b, c, alpha, beta, gamma = params
    alpha_rad = alpha  
    beta_rad = beta    
    gamma_rad = gamma  
    
    lattice_mat = lattice_matrix(a, b, c, alpha_rad, beta_rad, gamma_rad)
    L_prove.append(lattice_mat)

def find_nearest_fraction(value, fractions):
    min_diff = float('inf')
    closest_fraction = None
    for fraction in fractions:
        num, denom = map(int, fraction.split('/'))
        fraction_value = num / denom
        diff = abs(value - fraction_value)
        if diff < min_diff:
            min_diff = diff
            closest_fraction = fraction_value
    if min_diff <= 0.01:
        return closest_fraction
    else:
        return value


XYZ_decoded = arrays[2]
XYZ_processed = np.zeros_like(XYZ_decoded)

# fractions list
fractions = [
    '1/12', '1/2', '1/3', '1/4', '1/6', '1/8', '11/12', '11/8', 
    '13/12', '17/12', '2/3', '3/4', '3/8', '5/12', '5/4', '5/6', 
    '5/8', '7/12', '7/6', '7/8', '9/8'
]


for i in range(XYZ_decoded.shape[0]):  
    for j in range(XYZ_decoded.shape[2]):  
        for k in range(XYZ_decoded.shape[3]):  
            XYZ_processed[i, 0, j, k] = 0 if XYZ_decoded[i, 0, j, k] < 0.01 else find_nearest_fraction(XYZ_decoded[i, 0, j, k], fractions)

XYZ_processed = XYZ_processed.reshape(num_batches, 24, 3)

import numpy as np

def validate_data(XYZ_processed_list, A_processed_list, L_processed_list):
    cleaned_XYZ_list = []
    cleaned_A_list = []
    cleaned_L_list = []
    
    for XYZ_processed, A_processed, L_processed in zip(XYZ_processed_list, A_processed_list, L_processed_list):
        if set(np.unique(A_processed)) <= {0, 1}:
            continue
        
        cleaned_XYZ_data = []
        valid_group = True  
        
        for i, xyz in enumerate(XYZ_processed):
            if np.allclose(xyz, [0, 0, 0], atol=1e-8):
                if i == 0 or (i > 0 and not np.allclose(XYZ_processed[i-1], [0, 0, 0], atol=1e-8) and 
                              (i == len(XYZ_processed) - 1 or not np.allclose(XYZ_processed[i+1], [0, 0, 0], atol=1e-8))):
                    cleaned_XYZ_data.append(xyz)
            else:
                cleaned_XYZ_data.append(xyz)
        
        for xyz in cleaned_XYZ_data:
            if not ((0.32 <= xyz[2] <= 0.68) or ((0 <= xyz[2] <= 0.25) and (0.75 <= xyz[2] <= 1))):
                valid_group = False
                break
        
        if valid_group:
            cleaned_XYZ_list.append(cleaned_XYZ_data)
            cleaned_A_list.append(A_processed)
            cleaned_L_list.append(L_processed)
    
    return (np.array(cleaned_XYZ_list, dtype=object), 
            np.array(cleaned_A_list, dtype=object), 
            np.array(cleaned_L_list, dtype=object))

XYZ_processed,A_processed,L_processed=validate_data(XYZ_processed,A_processed,L_processed)


def validate_data(XYZ_processed_list, L_processed_list, A_processed_list):
    cleaned_XYZ_list = []
    cleaned_L_list = []
    cleaned_A_list = []
    
    for XYZ_processed, L_processed, A_processed in zip(XYZ_processed_list, L_processed_list, A_processed_list):
        cleaned_data = []
        non_zero_count = 0  
        
        for data in XYZ_processed:
            if not np.allclose(data, [0, 0, 0], atol=1e-8):
                non_zero_count += 1

        if non_zero_count > 2:
            continue
        
        for i, data in enumerate(XYZ_processed):
            if np.allclose(data, [0, 0, 0], atol=1e-8):
                if i == 0 or (i > 0 and not np.allclose(XYZ_processed[i-1], [0, 0, 0], atol=1e-8) and (i == len(XYZ_processed) - 1 or not np.allclose(XYZ_processed[i+1], [0, 0, 0], atol=1e-8))):
                    cleaned_data.append(data)
            else:
                cleaned_data.append(data)
        
        cleaned_XYZ_list.append(np.array(cleaned_data))
        cleaned_L_list.append(L_processed)
        cleaned_A_list.append(A_processed)
    
    return cleaned_XYZ_list, cleaned_L_list, cleaned_A_list

XYZ_pre,L_pre,A_pre = validate_data(XYZ_processed,L_processed,A_processed)


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


import numpy as np


def replace_values_per_row(A, xenes, M, X, A_elements, B_elements, O_elements):
    for i in range(A.shape[0]):
        unique_values = np.unique(A[i, A[i] != 0])
        
        replacement_dict = {}
        
        if set(unique_values) == {1}:
            replacement_dict[1] = np.random.choice(xenes)
        elif set(unique_values) == {1, 2}:
            replacement_dict[1] = np.random.choice(M)
            replacement_dict[2] = np.random.choice(X)
        elif set(unique_values) == {1, 5}:
            replacement_dict[1] = np.random.choice(M)
            replacement_dict[5] = np.random.choice(X)
        elif set(unique_values) == {1, 2, 5}:
            replacement_dict[1] = np.random.choice(A_elements)
            replacement_dict[2] = np.random.choice(B_elements)
            replacement_dict[5] = np.random.choice(O_elements)
        
        for j in range(A.shape[1]):
            for key in replacement_dict:
                if A[i, j] == key:
                    A[i, j] = replacement_dict[key]
    
    return A


xenes = [5, 14, 15, 32, 50]
M = [22, 40, 72, 23, 41, 73, 24, 42, 74, 25, 43, 75, 26, 27, 28, 29, 30]
X = [16, 34, 52]
A_elements = [38, 56, 57]  # Sr (38), Ba (56), La (57)
B_elements = [22, 25, 39, 40, 42]  # Ti (22), Mn (25), Zr (40), Mo (42)
O_elements = [8]


A_processed = replace_values_per_row(A_processed, xenes, M, X, A_elements, B_elements, O_elements)
print(A_processed)




import csv
import numpy as np
import json


def read_wyckoff_positions(file_path):
    wyckoff_data = {}
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            space_group = int(row[0])
            wyckoff_positions = eval(row[1])
            wyckoff_data[space_group] = wyckoff_positions
    return wyckoff_data

def get_wyckoff_positions(wyckoff_data, space_group):
    return wyckoff_data.get(space_group, [])

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def validate_and_clean_input_data(XYZ_processed):
    cleaned_data = []
    for i, data in enumerate(XYZ_processed):
        if np.allclose(data, [0, 0, 0], atol=1e-8):
            if i == 0 or (i > 0 and not np.allclose(XYZ_processed[i-1], [0, 0, 0], atol=1e-8) and (i == len(XYZ_processed) - 1 or not np.allclose(XYZ_processed[i+1], [0, 0, 0], atol=1e-8))):
                cleaned_data.append(data)
        else:
            cleaned_data.append(data)
    return np.array(cleaned_data)



def atomic_numbers_to_elements(array, element_list):

    array = np.array(array)
    
    if array.ndim == 1:
        array = array.reshape(1, -1)
    element_symbols = []
    for row in array:
        element_row = [
            element_list[num-1] if 0 < num <= len(element_list) else '0'
            for num in row
        ]
        element_symbols.append(element_row)
    
    return element_symbols


def check_and_replace(data, pos_list, num_indices, tolerance):
    replaced_data = []
    first_pos_values = pos_list[0].split(', ')
    if np.allclose(data[num_indices], [float(first_pos_values[i]) for i in num_indices], atol=tolerance):
        for pos in pos_list:
            new_position = [eval(x, {'x': data[0], 'y': data[1], 'z': data[2]}) for x in pos.split(', ')]
            replaced_data.append(new_position)
        return replaced_data, True
    return replaced_data, False

def compare_and_replace(XYZ_processed, A_processed, wyckoff_positions, tolerance=0.01):
    replaced_data = []
    new_A_processed = []

    for i, data in enumerate(XYZ_processed):
        replaced = False
        a_value = A_processed[i]

        for pos_list in wyckoff_positions:
            first_pos_values = pos_list[0].split(', ')

            if all(is_number(val) for val in first_pos_values):
                pos_eval = np.array([float(val) for val in first_pos_values])
                if np.allclose(data, pos_eval, atol=tolerance):
                    replaced_data.append(data.tolist())
                    new_A_processed.append(a_value)
                    replaced = True
                    break

        if replaced:
            continue

        for pos_list in wyckoff_positions:
            first_pos_values = pos_list[0].split(', ')

            if sum(is_number(val) for val in first_pos_values) == 2:
                num_indices = [i for i, val in enumerate(first_pos_values) if is_number(val)]
                new_replaced_data, replaced = check_and_replace(data, pos_list, num_indices, tolerance)
                if replaced:
                    replaced_data.extend(new_replaced_data)
                    new_A_processed.extend([a_value] * len(new_replaced_data))
                    break

        if replaced:
            continue

        for pos_list in wyckoff_positions:
            first_pos_values = pos_list[0].split(', ')

            if sum(is_number(val) for val in first_pos_values) == 1:
                num_index = [i for i, val in enumerate(first_pos_values) if is_number(val)][0]
                new_replaced_data, replaced = check_and_replace(data, pos_list, [num_index], tolerance)
                if replaced:
                    replaced_data.extend(new_replaced_data)
                    new_A_processed.extend([a_value] * len(new_replaced_data))
                    break

        if replaced:
            continue

        first_pos = wyckoff_positions[0][0]
        x, y, z = data
        new_positions = []
        for pos in wyckoff_positions[0]:
            new_x, new_y, new_z = pos.split(', ')
            new_positions.append([
                eval(new_x, {}, {'x': x, 'y': y, 'z': z}),
                eval(new_y, {}, {'x': x, 'y': y, 'z': z}),
                eval(new_z, {}, {'x': x, 'y': y, 'z': z})
            ])
        replaced_data.extend(new_positions)
        new_A_processed.extend([a_value] * len(new_positions))

    replaced_data = np.array(replaced_data)

    replaced_data[replaced_data < 0] += 1

    unique_data, indices = np.unique(replaced_data, axis=0, return_index=True)
    unique_A_processed = [new_A_processed[idx] for idx in indices]

    return unique_data, unique_A_processed

def process_input(XYZ_data, A_data, epsilon=1e-8):
    def is_unique(array):
        seen = []
        for triplet in array:
            is_duplicate = any(np.all(np.abs(triplet - seen_triplet) < epsilon) for seen_triplet in seen)
            if is_duplicate:
                return False
            seen.append(triplet)
        return True

    def remove_duplicates(array, a_array):
        unique_array = []
        unique_a_array = []
        seen = []
        for i, triplet in enumerate(array):
            is_duplicate = any(np.all(np.abs(triplet - seen_triplet) < epsilon) for seen_triplet in seen)
            if not is_duplicate:
                unique_array.append(triplet)
                unique_a_array.append(a_array[i])
                seen.append(triplet)
        return np.array(unique_array), np.array(unique_a_array)
    
    if not is_unique(XYZ_data):
        XYZ_data, A_data = remove_duplicates(XYZ_data, A_data)
    
    return XYZ_data, A_data

def generate_G_list(data):
    angle_90_rad = np.pi / 2
    angle_120_rad = 2 * np.pi / 3
    all_lists = []
    counts = []

    for row in data:
        if len(row) != 6:
            raise ValueError("Each row in the input data must have six elements: a, b, c, alpha, beta, gamma")
        a, b, c, alpha, beta, gamma = [np.round(x, 1) for x in row]
        G_list = []
        if a == b == c and np.allclose([alpha, beta, gamma], angle_90_rad, atol=0.1):
            G_list.extend([195,200,207,215,221])  # Cubic
        elif a == b and np.allclose([alpha, beta], angle_90_rad, atol=0.1) and np.isclose(gamma, angle_120_rad, atol=0.1):
            G_list.extend([143,144,145,146,147,150,156,157,164,168,174,175,183,187,189,191])  # Hexagonal
        elif a == b and np.allclose([alpha, beta, gamma], angle_90_rad, atol=0.1):
            G_list.extend([75,81,83,85,89,90,99,100,111,113,115,117,123,125,127,129])   # Tetragonal
        elif np.allclose([alpha, beta, gamma], angle_90_rad, atol=0.1):
            G_list.extend([16,18,21,25,28,32,35,47,50,51,55,59,65,67])    # Orthorhombic
        elif np.isclose(alpha, angle_90_rad, atol=0.1) and np.isclose(gamma, angle_90_rad, atol=0.1):
            G_list.extend([3,4,5,6,8,10,11,12])     # Monoclinic
        else:
            G_list.extend(range(1, 3))      # Triclinic
        
        all_lists.append(G_list)
        counts.append(len(G_list))

    return all_lists, counts


wyckoff_data = read_wyckoff_positions('../preprocess/wyckoff_list.csv')

XYZ_list=[]
A_list=[]
L_list=[]


all_G_lists=[]
all_counts=[]
Labc_list=[]
for i in range(len(L_pre)):
    G_list, counts = generate_G_list(L_pre[i].reshape(1, -1))
    all_G_lists.append(G_list)
    all_counts.append(counts)
    count = counts[0]  
    Labc_list.extend([L_processed[i]] * count*num_batches)

input_array = Labc_list

for params in input_array:
    a, b, c, alpha, beta, gamma = params
    alpha_rad = alpha  
    beta_rad = beta    
    gamma_rad = gamma  
    lattice_mat = lattice_matrix(a, b, c, alpha_rad, beta_rad, gamma_rad)
    L_list.append(lattice_mat)




for i, G_list in enumerate(all_G_lists):
    for sublist in G_list:
        for space_group in sublist:
            wyckoff_positions = get_wyckoff_positions(wyckoff_data, space_group)
            print(f"Group {i+1}, Space Group {space_group} 的Wyckoff Positions: {wyckoff_positions}")

            for i in range(len(XYZ_pre)):
                xyz=XYZ_pre[i]
                atomtype=A_pre[i]
                cleaned_XYZ_processed_list = validate_and_clean_input_data(xyz)
                updated_XYZ_processed, updated_A_processed = compare_and_replace(cleaned_XYZ_processed_list,atomtype, wyckoff_positions)
                processed_XYZ, processed_A = process_input(updated_XYZ_processed, updated_A_processed)
                converted_A_list = atomic_numbers_to_elements(processed_A, element_list)
                XYZ_list.append(processed_XYZ)
                A_list.append(converted_A_list)

import os
output_dir = "PS_WAXYZ_result"
os.makedirs(output_dir, exist_ok=True)


for i, (elements_row, lattice_vectors, coordinates) in enumerate(zip(A_list, L_list, XYZ_list)):
    zero_indices = np.where(elements_row == '0')[0]
    zero_index = zero_indices[0] if zero_indices.size > 0 else len(elements_row)
    elements = elements_row[:zero_index]
    elements_set, counts = np.unique(elements, return_counts=True)

    poscar_content = f"POSCAR {i}\n"
    poscar_content += "1.0\n"  
    for vector in lattice_vectors:
        poscar_content += "  ".join(f"{v:.16f}" for v in vector) + "\n"
    poscar_content += "  ".join(elements_set) + "\n"
    poscar_content += "  ".join(map(str, counts)) + "\n"
    poscar_content += "Direct\n"
    for coord_row in coordinates:
        poscar_content += "  ".join(map(str, coord_row)) + "\n"

    filename = os.path.join(output_dir, f"POSCAR_{i}")
    with open(filename, "w") as f:
        f.write(poscar_content)

    print(f"{filename} files processed")

    filename = f"POSCAR_{i}"
    poscar_path = os.path.join(output_dir, filename)

    if os.path.exists(poscar_path):
        try:
            structure = Structure.from_file(poscar_path)
            
            if len(structure) < 3:
                print(f'File {filename} has fewer than 3 atoms and was not moved.')
                continue
            
            all_bonds_valid = True
            
            for i in range(len(structure)):
                for j in range(i+1, len(structure)):
                    distance = structure.distance_matrix[i, j]
                    if distance < 1.5:
                        all_bonds_valid = False
                        break
            
            if all_bonds_valid:
                good_folder_path = '../SCDM/PS_WAXYZ_result/GOOD'  
                if not os.path.exists(good_folder_path):
                    os.makedirs(good_folder_path)
                
                good_file_path = os.path.join(good_folder_path, filename)
                os.rename(poscar_path, good_file_path)
                print(f'File {filename} has been moved to {good_folder_path} because all bond lengths are >= 1.5 Å and the total number of atoms is >= 3.')
            else:
                print(f'File {filename} has bond lengths less than 1.5 Å and was not moved.')
        except Exception as e:
            print(f'An error occurred while processing {filename}: {e}')


import shutil
import os
def group_files_by_80(src_dir):
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    if not files:
        print("NO files here")
        return
    file_counter = 0
    folder_counter = 1
    
    os.makedirs(os.path.join(src_dir, str(folder_counter)), exist_ok=True)
    
    for file in files:
        dest_folder = os.path.join(src_dir, str(folder_counter))
        if file_counter >= 80:
            folder_counter += 1
            file_counter = 0
            dest_folder = os.path.join(src_dir, str(folder_counter))
            os.makedirs(dest_folder, exist_ok=True)
        
        shutil.move(os.path.join(src_dir, file), os.path.join(dest_folder, file))
        file_counter += 1

src_directory = "../PS_WAXYZ_result/GOOD"
group_files_by_80(src_directory)