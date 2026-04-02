"""数据加载和处理工具函数"""
import os
import numpy as np
import pandas as pd
import warnings

def load_participants_metadata(file_path):
    """
    加载被试人口学信息
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        pd.DataFrame: 包含participant_id, group, Site, Age, Gender的DataFrame
    """
    df = pd.read_csv(file_path)
    if 'ScanDir ID' in df.columns:
        id_col = 'ScanDir ID'
    elif 'participant_id' in df.columns:
        id_col = 'participant_id'
    else:
        id_col = [c for c in df.columns if 'ID' in c.upper()][0]
        
    df['participant_id'] = df[id_col].astype(str).str.replace('sub-', '', regex=False)
    
    if 'DX' in df.columns:
        # 0 = Control (label 0), !=0 = ADHD (label 1)
        df['group'] = df['DX'].apply(lambda x: 0 if x == 0 else 1)
    else:
        raise ValueError("Cannot find DX column.")
        
    # 标准化列名
    for col in ['Site', 'Age', 'Gender', 'Handedness']:
        if col not in df.columns:
            for c in df.columns:
                if c.lower() == col.lower():
                    df.rename(columns={c: col}, inplace=True)
                    break
                    
    data_dict = {
        'participant_id': df['participant_id'],
        'group': df['group'],
        'Site': df['Site'].astype(str) if 'Site' in df.columns else 'Unknown',
        'Age': pd.to_numeric(df['Age'], errors='coerce') if 'Age' in df.columns else 0,
        'Gender': pd.to_numeric(df['Gender'], errors='coerce') if 'Gender' in df.columns else 0
    }
    
    df_clean = pd.DataFrame(data_dict).dropna(subset=['participant_id', 'group', 'Site'])
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].mean())
    df_clean['Gender'] = df_clean['Gender'].fillna(0)
    return df_clean


def find_all_fc_files(base_dir):
    """
    递归查找目录下所有FC矩阵文件
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        dict: {subject_id: file_path}
    """
    fc_files = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(('.txt', '.tsv', '.csv')):
                base_name = f.split('_')[0].replace('sub-', '')
                fc_files[base_name] = os.path.join(root, f)
    return fc_files


def fisher_z_transform(r):
    """
    Fisher's Z变换（用于相关系数）
    
    Args:
        r: 相关系数或数组
        
    Returns:
        变换后的值
    """
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))


def load_fc_matrix(file_path, expected_shape=(116, 116)):
    """
    加载单个FC矩阵文件
    
    Args:
        file_path: FC矩阵文件路径
        expected_shape: 期望的矩阵形状，默认(116, 116)
        
    Returns:
        np.ndarray or None: 加载的矩阵，如果失败返回None
    """
    try:
        fc_matrix = pd.read_csv(file_path, sep=None, header=None, engine='python').values
        if fc_matrix.shape != expected_shape:
            return None
        return fc_matrix
    except Exception as e:
        warnings.warn(f"Error loading {file_path}: {e}")
        return None


def fc_to_feature_vector(fc_matrix, use_fisher_z=True):
    """
    将FC矩阵转换为特征向量（上三角部分）
    
    Args:
        fc_matrix: FC矩阵
        use_fisher_z: 是否应用Fisher Z变换
        
    Returns:
        np.ndarray: 特征向量
    """
    iu = np.triu_indices_from(fc_matrix, k=1)
    fc_vector = fc_matrix[iu]
    
    if use_fisher_z:
        fc_vector = fisher_z_transform(fc_vector)
    
    return fc_vector