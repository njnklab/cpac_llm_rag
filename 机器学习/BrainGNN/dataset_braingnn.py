import os
import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import Dataset, Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. BrainGNN requires PyG.")
    # Dummy classes for syntax checking
    class Dataset:
        pass
    class Data:
        pass


class BrainGNNDataset(Dataset):
    """
    Dataset for BrainGNN model.
    
    Converts FC matrices to graph structure with:
    - Node features: Each node's connectivity pattern (row of FC matrix)
    - Edge index: Fully connected graph (no self-loops)
    - Edge weights: FC connectivity values
    - Demographic features: Gender, Age, Handedness (optional)
    """
    
    def __init__(self, fc_paths, labels, subject_ids, demographic_features=None, n_nodes=116, 
                 diagonal_zero=True, use_abs_weight=False, transform=None,
                 add_noise=False, noise_std=0.01):
        """
        Args:
            fc_paths: List of paths to FC matrix files
            labels: List of labels (0/1)
            subject_ids: List of subject IDs
            demographic_features: Array of shape [N, D] containing demographic info
            n_nodes: Number of nodes (ROIs), default 116
            diagonal_zero: Whether to set diagonal to zero
            use_abs_weight: Whether to use absolute values for edge weights
            transform: Optional transform to apply
            add_noise: Whether to add noise to FC matrix
            noise_std: Standard deviation of noise
        """
        super(BrainGNNDataset, self).__init__(root=None, transform=transform)
        self.fc_paths = fc_paths
        self.labels = labels
        self.subject_ids = subject_ids
        self.demographic_features = demographic_features
        self.n_nodes = n_nodes
        self.diagonal_zero = diagonal_zero
        self.use_abs_weight = use_abs_weight
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        # Pre-build edge index (same for all graphs)
        self.edge_index = self._build_edge_index()
        
    def _build_edge_index(self):
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def len(self):
        return len(self.fc_paths)
    
    def get(self, idx):
        fc_path = self.fc_paths[idx]
        label = self.labels[idx]
        subject_id = self.subject_ids[idx]
        
        # Load FC matrix
        fc_matrix = self._load_fc_matrix(fc_path)
        
        if np.isnan(fc_matrix).any():
            fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)
            
        if self.add_noise:
            noise = np.random.normal(0, self.noise_std, fc_matrix.shape)
            fc_matrix = fc_matrix + noise
            fc_matrix = np.clip(fc_matrix, -1.0, 1.0)
        
        edge_fc = fc_matrix.copy()
        if self.diagonal_zero:
            np.fill_diagonal(fc_matrix, 0)
            np.fill_diagonal(edge_fc, 0)
        
        fc_matrix = fc_matrix.astype(np.float32)
        f_min, f_max = fc_matrix.min(), fc_matrix.max()
        if f_max > f_min:
            x_norm = (fc_matrix - f_min) / (f_max - f_min)
        else:
            x_norm = fc_matrix
        x = torch.tensor(x_norm, dtype=torch.float32)
        
        edge_weight = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    w = edge_fc[i, j]
                    if self.use_abs_weight:
                        w = abs(w)
                    edge_weight.append(w)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        y = torch.tensor(label, dtype=torch.long)
        
        data = Data(x=x, edge_index=self.edge_index, edge_attr=edge_weight, y=y)
        data.subject_id = subject_id
        
        # Add demographic features if available
        if self.demographic_features is not None:
            demo = self.demographic_features[idx]
            data.demo = torch.tensor(demo, dtype=torch.float32).unsqueeze(0) # [1, D]
        
        return data
    
    def _load_fc_matrix(self, fc_path):
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
    
    Returns:
        fc_paths: List of FC file paths
        labels: List of labels
        subject_ids: List of subject IDs
        demographics: Numpy array [N, 3] (gender, age, handedness)
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



class GraphFeatureNormalizer:
    """
    Normalization transforms for graph node features and edge weights.
    """
    
    @staticmethod
    def z_score_normalize_node_features(data):
        """Z-score normalization of node features."""
        x = data.x
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        data.x = (x - mean) / (std + 1e-8)
        return data
    
    @staticmethod
    def z_score_normalize_edge_weights(data):
        """Z-score normalization of edge weights."""
        edge_attr = data.edge_attr
        mean = edge_attr.mean()
        std = edge_attr.std()
        data.edge_attr = (edge_attr - mean) / (std + 1e-8)
        return data
    
    @staticmethod
    def normalize_edge_weights(data):
        """Normalize edge weights to [0, 1]."""
        edge_attr = data.edge_attr
        min_val = edge_attr.min()
        max_val = edge_attr.max()
        data.edge_attr = (edge_attr - min_val) / (max_val - min_val + 1e-8)
        return data


if __name__ == "__main__":
    if not PYG_AVAILABLE:
        print("Cannot test without PyTorch Geometric")
        exit(0)
    
    # Test dataset loading
    participants_path = '/mnt/sda1/zhangyan/cpac_output/机器学习/人口学/ds002748_participants.tsv'
    fc_dir = '/mnt/sda1/zhangyan/cpac_output/feature/deepprep/ds002748/'
    
    if os.path.exists(participants_path) and os.path.exists(fc_dir):
        fc_paths, labels, subject_ids = load_fc_data(
            participants_path, fc_dir, n_nodes=116
        )
        
        # Create dataset
        dataset = BrainGNNDataset(fc_paths, labels, subject_ids, n_nodes=116)
        
        # Test loading
        if len(dataset) > 0:
            data = dataset[0]
            print(f"\nSample loaded:")
            print(f"  Node features shape: {data.x.shape}")
            print(f"  Edge index shape: {data.edge_index.shape}")
            print(f"  Edge weight shape: {data.edge_attr.shape}")
            print(f"  Label: {data.y.item()}")
            print(f"  Subject ID: {data.subject_id}")
            print(f"  Node feature range: [{data.x.min():.4f}, {data.x.max():.4f}]")
            print(f"  Edge weight range: [{data.edge_attr.min():.4f}, {data.edge_attr.max():.4f}]")
    else:
        print("Data paths not found, skipping test")
