# SCDM：Discovery of 2D Materials via Symmetry-Constrained Diffusion Model
We introduce a symmetry-constrained diffusion model (SCDM) that integrates space group symmetry into the generative process.

![图片1](https://github.com/user-attachments/assets/1608e7f2-da93-44bb-8002-bd5070ece32c)



# Environemnt
You can run !pip install -r requirement.txt to install packages required.

# Usage

FIRST Step:preprocess
1.Using npz.py for data preprocessing via wyckoff position.
2.Change the path of the csv and poscar data
3.Generate the xxx.npz of your dataset

Tips: All the 2d-material data used by this work is in the ../SCDM/poscar

Second Step:train the diffusion model
1.Using  python train.py 

