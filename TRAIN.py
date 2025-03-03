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

#your npz data here
dataname='xxx.npz'
data=np.load('xxxx.npz')
x=data['x']

x=torch.Tensor(x).to(torch.float)#.cuda()
print('x',type(x))

#your poscar csv here
csv_file = '../xxx.csv'


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
epoches = 800

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

#train
DDPM = Trainer(DDPM, model_save_path=saved_path)

