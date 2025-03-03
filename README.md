# SCDM
2D Crystal generation via diffusion models guided by symmetry group constraints.

#Environemnt
You can run !pip install -r requirement.txt to install packages required.

#Usage


FIRST Step:preprocess
1.Using npz.py for data preprocessing via wyckoff position.
2.Change the path of the csv and poscar data
3.Generate the xxx.npz of your dataset

Tips: All the 2d-material data used by this work is in the ../SCDM/poscar

Second Step:train the diffusion model
1.Using  python train.py 

