# SCDM：Discovery of 2D Materials via Symmetry-Constrained Diffusion Model


🔬 A symmetry-constrained diffusion model (SCDM) that integrates space group symmetry into the generative process for 2D material discovery.

![图片1](https://github.com/user-attachments/assets/1608e7f2-da93-44bb-8002-bd5070ece32c)


## ✨ Key Features
- **Symmetry-aware Generation**: Incorporates space group symmetry constraints in diffusion process
- **Wyckoff Position Processing**: Automated data preprocessing for crystallographic data
- **2D Material Focus**: Specialized for two-dimensional material discovery

## 🚀 Getting Started

### Prerequisites
- Python 3.9+  
- PyTorch 2.0+  
- CUDA-enabled GPU (recommended cu121) 

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/SCDM.git

# Install dependencies
pip install -r requirements.txt
```

# 📖 Usage
## Step 1: Data Preprocessing
0.Using csv_create.py for coding all the data that you used

1.Prepare your crystallographic data in cif format under```bash ..SCDM/poscar/```

2.Modify file paths in npz.py
```bash
# your data csv here
csv_file = '../SCDM/POSCAR/xxx.csv'
# your poscar data here
directory_path = '../POSCAR'
```
3.Generate processed dataset:
```bash python npz.py --output materials_dataset.npz```
Tips: All the 2d-material data used by this work is in the ../SCDM/poscar

## Step 2: Model training
RUN 
```bash
python train.py\
    --dataset materials_dataset.npz 
    --batch_size 24 
    --num_epochs 100 
    --save_dir models
```


Second Step:train the diffusion model
1.Using  python train.py 

