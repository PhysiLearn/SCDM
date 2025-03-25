# 🔬 SCDM：Discovery of 2D Materials via Symmetry-Constrained Diffusion Model

A symmetry-constrained diffusion model (SCDM) that integrates space group symmetry into the generative process for 2D material discovery.

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
0.Prepare your crystallographic data in cif format under``` ..SCDM/poscar/```

1.Use ```python csv_create.py ``` for encoding all the data  under ``` ..SCDM/poscar/```

2.Modify file paths in **npz.py**
```
# your data csv here
csv_file = '../SCDM/POSCAR/xxx.csv'
# your poscar data here
directory_path = '../POSCAR'
```
3.Generate processed dataset:
``` python npz.py --output materials_dataset.npz```

## Step 2: Model training

```
python TRAIN.py\
    --dataset materials_dataset.npz 
    --batch_size 24 
    --num_epochs 100 
    --save_dir models
```
## Step 3: Generation 
0. Load Your Trained Model
before generating new materials, load your trained model:
```
best_model_path = saved_path + '/' + 'BestModel.pth'
#your trained model here
best_model_path ='../saved_train_models/BestModel.pth'
DDPM.load_state_dict(torch.load(best_model_path)) 
```
1.Run the generation script to create new 2D materials:
```
python Generate.py

```

# 🤝 Citation
```
@article{xu2024discovery,
  title={Discovery of 2D Materials via Symmetry-Constrained Diffusion Model},
  author={Xu, Shihang and Chu, Shibing and Mrad, Rami and Zhang, Zhejun and Li, Zhelin and Jiao, Runxian and Chen, Yuanping},
  journal={arXiv preprint arXiv:2412.18414},
  year={2024}
}
```

# 📜 License
This project is licensed under the MIT License.

# 📞 Contact
If there are any issues and questions, please leave us issues on Github or send me email (Shibing Chu c◎ujs dot edu dot cn).


