import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BrainNetCNNDataset(Dataset):
    """
    Dataset for BrainNetCNN model.
    
    Loads FC matrices and demographic features (Gender, Age, Handedness).
    """
    
    def __init__(self, fc_paths, labels, subject_ids, demographic_features=None, n_nodes=116, 
                 diagonal_zero=True, transform=None, add_noise=False, noise_std=0.01):
        """
        Args:
            fc_paths: List of paths to FC matrix files
            labels: List of labels (0/1)
            subject_ids: List of subject IDs
            demographic_features: Array [N, D] of demographic info
            n_nodes: Number of nodes (ROIs), default 116
            diagonal_zero: Whether to set diagonal to zero
            transform: Optional transform to apply
        """
        self.fc_paths = fc_paths
        self.labels = labels
        self.subject_ids = subject_ids
        self.demographic_features = demographic_features
        self.n_nodes = n_nodes
        self.diagonal_zero = diagonal_zero
        self.transform = transform
        self.add_noise = add_noise
        self.noise_std = noise_std
        
    def __len__(self):
        return len(self.fc_paths)
    
    def __getitem__(self, idx):
        fc_path = self.fc_paths[idx]
        label = self.labels[idx]
        subject_id = self.subject_ids[idx]
        
        # Load FC matrix
        fc_matrix = self._load_fc_matrix(fc_path)
        
        # Preprocess
        if self.diagonal_zero:
            np.fill_diagonal(fc_matrix, 0)
        
        # Convert to float32
        fc_matrix = fc_matrix.astype(np.float32)
        
        # Add Gaussian Noise for data augmentation (Scheme B)
        if self.add_noise:
            noise = np.random.normal(0, self.noise_std, fc_matrix.shape)
            fc_matrix = fc_matrix + noise
            fc_matrix = np.clip(fc_matrix, -1.0, 1.0)

        # Add channel dimension: [N, N] -> [1, N, N]
        fc_tensor = torch.tensor(fc_matrix, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            fc_tensor = self.transform(fc_tensor)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Demographic features
        if self.demographic_features is not None:
            demo = self.demographic_features[idx]
            demo_tensor = torch.tensor(demo, dtype=torch.float32)
            return fc_tensor, label_tensor, subject_id, demo_tensor
        
        return fc_tensor, label_tensor, subject_id
    
    def _load_fc_matrix(self, fc_path):
        """Load FC matrix from file."""
        if fc_path.endswith('.txt'):
            fc_matrix = np.loadtxt(fc_path)
        else:
            fc_matrix = pd.read_csv(fc_path, sep=None, header=None, engine='python').values
        
        if fc_matrix.shape[0] != fc_matrix.shape[1]:
            raise ValueError(f"FC matrix must be square, got shape {fc_matrix.shape}")
        
        if fc_matrix.shape[0] != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes}x{self.n_nodes} matrix, got {fc_matrix.shape}")
        
        return fc_matrix


def load_participants_metadata(file_path):
    """加载并标准化被试元数据，包括性别、年龄、利手和诊断。"""
    if file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)

    # 1. 标准化 ID 列名
    if 'participant_id' in df.columns:
        id_col = 'participant_id'
    elif 'ScanDir ID' in df.columns:
        id_col = 'ScanDir ID'
    else:
        id_col = [c for c in df.columns if 'ID' in c.upper()][0]
    
    # 2. 标准化 分组标签 (DX)
    if 'group' in df.columns:
        if df['group'].dtype == object:
            mapping = {'control': 0, 'depr': 1, 'healthy': 0, 'patient': 1}
            df['group_mapped'] = df['group'].str.lower().map(mapping)
        else:
            df['group_mapped'] = df['group']
    elif 'DX' in df.columns:
        df['group_mapped'] = df['DX'].apply(lambda x: 0 if x == 0 else 1)
    else:
        raise ValueError(f"无法在文件 {file_path} 中找到 'group' 或 'DX' 列")

    # 3. 提取 Gender, Age, Handedness
    # Gender
    if 'Gender' in df.columns:
        df['gender_val'] = df['Gender']
    elif 'gender' in df.columns:
        df['gender_val'] = df['gender'].map({'m': 0, 'f': 1, 'M': 0, 'F': 1})
    else:
        df['gender_val'] = 0
    
    # Age
    if 'Age' in df.columns:
        df['age_val'] = df['Age']
    elif 'age' in df.columns:
        df['age_val'] = df['age']
    else:
        df['age_val'] = 0
        
    # Handedness
    if 'Handedness' in df.columns:
        df['hand_val'] = df['Handedness']
    elif 'Edinburgh' in df.columns:
        df['hand_val'] = pd.to_numeric(df['Edinburgh'], errors='coerce')
    else:
        df['hand_val'] = 1 # Default right-handed
        
    # 填充缺失值
    df['gender_val'] = df['gender_val'].fillna(0)
    df['age_val'] = pd.to_numeric(df['age_val'], errors='coerce').fillna(df['age_val'].mean() if 'age_val' in df and not df['age_val'].isna().all() else 0)
    df['hand_val'] = df['hand_val'].fillna(1)

    data_dict = {
        'participant_id': df[id_col].astype(str).str.replace('sub-', '', regex=False),
        'group': df['group_mapped'],
        'gender': df['gender_val'],
        'age': df['age_val'],
        'handedness': df['hand_val']
    }
    df_clean = pd.DataFrame(data_dict).dropna(subset=['group'])
    return df_clean

def load_fc_data(participants_csv, fc_dir, n_nodes=116):
    """
    Load FC data and labels from participants CSV and FC directory.
    """
    participants_df = load_participants_metadata(participants_csv)
    
    fc_files = [
        f for f in os.listdir(fc_dir)
        if f.endswith(('.tsv', '.txt', '.csv'))
    ]
    
    def get_sid(fname):
        raw_id = fname.split('_')[0]
        return raw_id.replace('sub-', '')
    
    fc_file_map = {get_sid(f): os.path.join(fc_dir, f) for f in fc_files}
    
    # Map back to participants
    fc_paths = []
    labels = []
    subject_ids = []
    demographics = []
    
    for _, row in participants_df.iterrows():
        sid = row['participant_id']
        if sid in fc_file_map:
            fc_paths.append(fc_file_map[sid])
            labels.append(row['group'])
            subject_ids.append(sid)
            demographics.append([row['gender'], row['age'], row['handedness']])
    
    print(f"Successfully loaded {len(fc_paths)} samples")
    return fc_paths, labels, subject_ids, np.array(demographics, dtype=np.float32)



class FCNormalizer:
    """
    Normalization transforms for FC matrices.
    """
    
    @staticmethod
    def z_score_normalize(fc_tensor):
        """Z-score normalization across all elements."""
        mean = fc_tensor.mean()
        std = fc_tensor.std()
        return (fc_tensor - mean) / (std + 1e-8)
    
    @staticmethod
    def min_max_normalize(fc_tensor):
        """Min-max normalization to [0, 1]."""
        min_val = fc_tensor.min()
        max_val = fc_tensor.max()
        return (fc_tensor - min_val) / (max_val - min_val + 1e-8)
    
    @staticmethod
    def fisher_z_transform(fc_tensor):
        """Apply Fisher z-transform (arctanh)."""
        # Clip to valid range for arctanh
        fc_tensor = torch.clamp(fc_tensor, -0.999, 0.999)
        return torch.arctanh(fc_tensor)


if __name__ == "__main__":
    # Test dataset loading
    participants_path = '/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/ds002748_participants.tsv'
    fc_dir = '/mnt/sda1/zhangyan/cpac_output/feature/deepprep/ds002748/'
    
    if os.path.exists(participants_path) and os.path.exists(fc_dir):
        fc_paths, labels, subject_ids = load_fc_data(
            participants_path, fc_dir, n_nodes=116
        )
        
        # Create dataset
        dataset = BrainNetCNNDataset(fc_paths, labels, subject_ids, n_nodes=116)
        
        # Test loading
        if len(dataset) > 0:
            fc_tensor, label, subject_id = dataset[0]
            print(f"\nSample loaded:")
            print(f"  FC tensor shape: {fc_tensor.shape}")
            print(f"  Label: {label.item()}")
            print(f"  Subject ID: {subject_id}")
            print(f"  FC value range: [{fc_tensor.min():.4f}, {fc_tensor.max():.4f}]")
    else:
        print("Data paths not found, skipping test")
