import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pyxtal import pyxtal
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from functools import partial
import multiprocessing
import os

import optax
import math


from preprocess.wyckoff import mult_table
from preprocess.elements import element_list

@jax.vmap
def sort_atoms(W, A, X):
    """
    lex sort atoms according W, X, Y, Z

    W: (n, )
    A: (n, )
    X: (n, dim) int
    """
    W_temp = jnp.where(W>0, W, 9999) # change 0 to 9999 so they remain in the end after sort

    X -= jnp.floor(X)
    idx = jnp.lexsort((X[:,2], X[:,1], X[:,0], W_temp))

    #assert jnp.allclose(W, W[idx])
    A = A[idx]
    X = X[idx]
    return A, X

def letter_to_number(letter):
    """
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27 
    """
    return ord(letter) - ord('a') + 1 if 'a' <= letter <= 'z' else 27 if letter == 'A' else None

def shuffle(key, data):
    """
    shuffle data along batch dimension
    """
    G, L, XYZ, A, W = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return G[idx], L[idx], XYZ[idx], A[idx], W[idx]
    
def process_one(cif, atom_types, wyck_types, n_max, tol=0.01):
    """
    # taken from https://anonymous.4open.science/r/DiffCSP-PP-8F0D/diffcsp/common/data_utils.py
    Process one cif string to get G, L, XYZ, A, W

    Args:
      cif: cif string
      atom_types: number of atom types
      wyck_types: number of wyckoff types
      n_max: maximum number of atoms in the unit cell
      tol: tolerance for pyxtal

    Returns:
      G: space group number
      L: lattice parameters
      XYZ: fractional coordinates
      A: atom types
      W: wyckoff letters
    """
     # 从给定的CIF字符串和格式中创建晶体结构对象
    crystal = Structure.from_str(cif, fmt='cif')
    # 使用SpacegroupAnalyzer分析晶体的空间群，并得到精炼后的晶体结构
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    # 初始化pyxtal库的晶体对象
    c = pyxtal()
    try:
        # 尝试使用默认的容差值生成晶体的种子
        c.from_seed(crystal, tol=0.01)
    except:
        # 如果默认容差值失败，则使用更小的容差值重试
        c.from_seed(crystal, tol=0.0001)
    # 获取晶体的对称性群编号
    g = c.group.number
     # 获取晶胞中的原子数目
    num_sites = len(c.atom_sites)
    # 确保n_max大于原子数目，以留出空间放置晶格参数
    print(f"Debug: n_max = {n_max}, num_sites = {num_sites}")
    assert (n_max > num_sites) # we will need at least one empty site for output of L params
    # 打印空间群编号、符号和原子数目
    #print ('空间群编号,符号,原子数目\n',g, c.group.symbol, num_sites)
    natoms = 0
    ww = []
    aa = []
    fc = []
    ws = []
    # 遍历晶体中的每个原子位置
    for site in c.atom_sites:
        # 获取原子类型索引
        a = element_list.index(site.specie) 
        # 获取原子的分数坐标
        x = site.position
        # 获取Wyckoff位置的多重性
        m = site.wp.multiplicity
        # 将Wyckoff字母转换为数字
        w = letter_to_number(site.wp.letter)
        # 构建Wyckoff位置的符号
        symbol = str(m) + site.wp.letter
        # 更新原子数目
        natoms += site.wp.multiplicity
        # 断言原子类型和Wyckoff位置的编号在允许的范围内
        assert (a < atom_types)
        assert (w < wyck_types)
        # 断言分数坐标与通过Wyckoff位置操作后的结果足够接近
        assert (np.allclose(x, site.wp[0].operate(x)))
        # 将原子类型、Wyckoff位置和分数坐标添加到列表中
        aa.append( a )
        ww.append( w )
        fc.append( x )  # the generator of the orbit
        ws.append( symbol )
        # 打印处理过程中的详细信息
        #print ('g, a, w, m, symbol, x:', g, a, w, m, symbol, x)
    # 使用Wyckoff位置编号对原子进行排序
    idx = np.argsort(ww)
    ww = np.array(ww)[idx]
    aa = np.array(aa)[idx]
    # 将分数坐标重塑为一个二维数组，每行代表一个原子的位置
    fc = np.array(fc)[idx].reshape(num_sites, 3)
    ws = np.array(ws)[idx]
    # 打印排序后的Wyckoff符号、原子类型和Wyckoff位置
    #print (ws, aa, ww, natoms) 
    # 如果原子数目小于n_max，则在列表末尾填充0
    aa = np.concatenate([aa,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)

    ww = np.concatenate([ww,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)
    # 在分数坐标列表末尾填充一个很大的数值，以表示空位
    fc = np.concatenate([fc, 
                         np.full((n_max - num_sites, 3), 1e10)],
                        axis=0)
    # 计算晶格参数的几何平均值
    abc = np.array([c.lattice.a, c.lattice.b, c.lattice.c])/natoms**(1./3.)
    # 获取晶格角度
    angles = np.array([c.lattice.alpha, c.lattice.beta, c.lattice.gamma])
    # 将晶格参数和晶格角度合并成晶格参数向量
    l = np.concatenate([abc, angles])
    
    
    #print ('===================================')
    # 返回计算出的空间群编号、晶格参数、原子分数坐标、原子类型和Wyckoff字母
    return g, l, fc, aa, ww 

def GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max, num_workers=1):
    """
    Read cif strings from csv file and convert them to G, L, XYZ, A, W
    Note that cif strings must be in the column 'cif'

    Args:
      csv_file: csv file containing cif strings
      atom_types: number of atom types
      wyck_types: number of wyckoff types
      n_max: maximum number of atoms in the unit cell
      num_workers: number of workers for multiprocessing

    Returns:
      G: space group number
      L: lattice parameters
      XYZ: fractional coordinates
      A: atom types
      W: wyckoff letters
    """
    data = pd.read_csv(csv_file)
    cif_strings = data['cif']

    p = multiprocessing.Pool(num_workers)
    partial_process_one = partial(process_one, atom_types=atom_types, wyck_types=wyck_types, n_max=n_max)
    results = p.map_async(partial_process_one, cif_strings).get()
    p.close()
    p.join()

    G, L, XYZ, A, W = zip(*results)
    

    G = jnp.array(G) 
    A = jnp.array(A).reshape(-1, n_max)
    W = jnp.array(W).reshape(-1, n_max)
    XYZ = jnp.array(XYZ).reshape(-1, n_max, 3)
    L = jnp.array(L).reshape(-1, 6)

    A, XYZ = sort_atoms(W, A, XYZ)
    
    return G, L, XYZ, A, W

def GLXA_to_structure_single(G, L, X, A):
    """
    Convert G, L, X, A to pymatgen structure. Do not use this function due to the bug in pymatgen.

    Args:
      G: space group number
      L: lattice parameters
      X: fractional coordinates
      A: atom types
    
    Returns:
      structure: pymatgen structure
    """
    lattice = Lattice.from_parameters(*L)
    # filter out padding atoms
    idx = np.where(A > 0)
    A = A[idx]
    X = X[idx]
    structure = Structure.from_spacegroup(sg=G, lattice=lattice, species=A, coords=X).as_dict()

    return structure

def GLXA_to_csv(G, L, X, A, num_worker=1, filename='out_structure.csv'):

    L = np.array(L)
    X = np.array(X)
    A = np.array(A)
    p = multiprocessing.Pool(num_worker)
    if isinstance(G, int):
        G = np.array([G] * len(L))
    structures = p.starmap_async(GLXA_to_structure_single, zip(G, L, X, A)).get()
    p.close()
    p.join()

    data = pd.DataFrame()
    data['cif'] = structures
    header = False if os.path.exists(filename) else True
    data.to_csv(filename, mode='a', index=False, header=header)


# if __name__ == '__main__':
#     atom_types = 119
#     wyck_types = 28
#     n_max = 24

#     import numpy as np 
#     np.set_printoptions(threshold=np.inf)
    
#     csv_file = '/home/l/Disk1/XUHang/diffusion/2DCrystalDiffusion/PCCD/preprocess/2024_5_14.csv'

#     G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max)

#     # 将处理后的数据转换为 pandas DataFrame
#     df_G = pd.DataFrame(G, columns=['G'])
#     df_L = pd.DataFrame(L, columns=['L'])  # 假设 L 是形状为 (-1, 6) 的数组
#     df_XYZ = pd.DataFrame(XYZ, columns=['XYZ'])  # 假设 XYZ 是形状为 (-1, 3) 的数组
#     df_A = pd.DataFrame(A, columns=['A'])
#     df_W = pd.DataFrame(W, columns=['W'])
    
#     original_df = pd.read_csv(csv_file)
#     final_df = pd.concat([original_df, df_G, df_L, df_XYZ, df_A, df_W], axis=1)

#     # 保存修改后的 DataFrame 回原始 CSV 文件
#     final_df.to_csv(csv_file, index=False)

#     print(f'Updated data has been saved back to {csv_file}')
    
    # print(type(G))
    # #print(len(G))
    # print(type(L))
    # #print(len(L))
    # print(type(XYZ))
    # #print(len(XYZ))
    # print(type(A))
    # #print(len(A))
    # print(type(W))
    # #print(len(W))

    #定义保存数据的文件名
    # output_file = 'data.txt'

    # # 打开文件用于写入
    # with open(output_file, 'w') as f:
    #     # 遍历data元组中的每个数组
    #     for array in data:
    #         # 将数组转换为字符串
    #         array_str = np.array2string(array, separator=',')
    #         # 写入到文件中，并在每个数组后添加换行符
    #         f.write(array_str + '\n')

    # print(f'Data has been saved to {output_file}')

