from rdkit import Chem
import pandas as pd
import os

# 定义提取信息的函数
def extract_info(mol):
    props = mol.GetPropsAsDict()
    info = {
        'PUBCHEM_CONFORMER_RMSD': props.get('PUBCHEM_CONFORMER_RMSD', 0),
        'PUBCHEM_EFFECTIVE_ROTOR_COUNT': props.get('PUBCHEM_EFFECTIVE_ROTOR_COUNT', 0),
        'PUBCHEM_HEAVY_ATOM_COUNT': props.get('PUBCHEM_HEAVY_ATOM_COUNT', 0),
        'PUBCHEM_ATOM_DEF_STEREO_COUNT': props.get('PUBCHEM_ATOM_DEF_STEREO_COUNT', 0),
        'PUBCHEM_ATOM_UDEF_STEREO_COUNT': props.get('PUBCHEM_ATOM_UDEF_STEREO_COUNT', 0),
        'PUBCHEM_BOND_DEF_STEREO_COUNT': props.get('PUBCHEM_BOND_DEF_STEREO_COUNT', 0),
        'PUBCHEM_BOND_UDEF_STEREO_COUNT': props.get('PUBCHEM_BOND_UDEF_STEREO_COUNT', 0),
        'PUBCHEM_ISOTOPIC_ATOM_COUNT': props.get('PUBCHEM_ISOTOPIC_ATOM_COUNT', 0),
        'PUBCHEM_COMPONENT_COUNT': props.get('PUBCHEM_COMPONENT_COUNT', 0),
        'PUBCHEM_CACTVS_TAUTO_COUNT': props.get('PUBCHEM_CACTVS_TAUTO_COUNT', 0),
        'PUBCHEM_MMFF94_ENERGY': props.get('PUBCHEM_MMFF94_ENERGY', 0),
        'PUBCHEM_FEATURE_SELFOVERLAP': props.get('PUBCHEM_FEATURE_SELFOVERLAP', 0),
        'PUBCHEM_SHAPE_SELFOVERLAP': props.get('PUBCHEM_SHAPE_SELFOVERLAP', 0),
        'PUBCHEM_SHAPE_VOLUME': props.get('PUBCHEM_SHAPE_VOLUME', 0)
    }
    return info

# 读取SDF文件夹中的所有文件
data_folder = 'data'
sdf_files = [f for f in os.listdir(data_folder) if f.endswith('.sdf')]

# 创建一个空DataFrame用于存储信息
df = pd.DataFrame(columns=['PUBCHEM_CONFORMER_RMSD', 'PUBCHEM_EFFECTIVE_ROTOR_COUNT', 'PUBCHEM_HEAVY_ATOM_COUNT',
                           'PUBCHEM_ATOM_DEF_STEREO_COUNT', 'PUBCHEM_ATOM_UDEF_STEREO_COUNT',
                           'PUBCHEM_BOND_DEF_STEREO_COUNT', 'PUBCHEM_BOND_UDEF_STEREO_COUNT',
                           'PUBCHEM_ISOTOPIC_ATOM_COUNT', 'PUBCHEM_COMPONENT_COUNT', 'PUBCHEM_CACTVS_TAUTO_COUNT',
                           'PUBCHEM_MMFF94_ENERGY', 'PUBCHEM_FEATURE_SELFOVERLAP', 'PUBCHEM_SHAPE_SELFOVERLAP',
                           'PUBCHEM_SHAPE_VOLUME'])

# 逐个读取SDF文件，并提取信息
for sdf_file in sdf_files:
    suppl = Chem.SDMolSupplier(os.path.join(data_folder, sdf_file))
    for mol in suppl:
        if mol:
            info = extract_info(mol)
            df = df.append(info, ignore_index=True)

# # 将DataFrame保存为CSV文件
# df.to_csv('y.csv', index=False)
y = df