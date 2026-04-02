"""
BrainGNN Model Implementation
=============================
Graph Neural Network for brain network analysis.
Uses PyTorch Geometric for graph operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GraphConv, GATConv, TopKPooling, global_mean_pool, global_max_pool
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. BrainGNN requires PyG.")


class GraphConvBlock(nn.Module):
    """
    Graph Convolution Block with BatchNorm, ReLU, and Dropout.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.5, conv_type='gcn'):
        super(GraphConvBlock, self).__init__()
        
        # Select convolution type
        if conv_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        elif conv_type == 'graph':
            self.conv = GraphConv(in_channels, out_channels)
        elif conv_type == 'gat':
            self.conv = GATConv(in_channels, out_channels, heads=1, concat=False)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [N, C_in]
            edge_index: Edge indices [2, E]
            edge_weight: Edge weights [E] (optional)
        Returns:
            x: Updated node features [N, C_out]
        """
        x = self.conv(x, edge_index, edge_weight)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class BrainGNN(nn.Module):
    """
    BrainGNN Model
    
    Graph Neural Network for brain network classification.
    Architecture: Graph Conv -> Graph Conv -> TopK Pool -> Graph Conv -> Readout -> MLP
    
    Args:
        in_channels: Number of input node features (default 116 for FC matrix rows)
        hidden_channels: Hidden dimension size
        num_classes: Number of output classes (default 2)
        pooling_ratio: Ratio of nodes to keep after pooling (default 0.5)
        dropout_rate: Dropout probability
        conv_type: Type of graph convolution ('gcn', 'graph', 'gat')
    """
    def __init__(self, in_channels=116, hidden_channels=64, num_classes=2, 
                 pooling_ratio=0.5, dropout_rate=0.5, conv_type='gcn', demo_dim=0):
        super(BrainGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.pooling_ratio = pooling_ratio
        self.demo_dim = demo_dim
        
        # Graph convolution layers
        self.conv1 = GraphConvBlock(in_channels, hidden_channels, dropout_rate, conv_type)
        self.conv2 = GraphConvBlock(hidden_channels, hidden_channels, dropout_rate, conv_type)
        
        # TopK Pooling: selects most important nodes
        self.pool = TopKPooling(hidden_channels, ratio=pooling_ratio)
        
        # Another conv after pooling
        self.conv3 = GraphConvBlock(hidden_channels, hidden_channels * 2, dropout_rate, conv_type)
        
        # Hybrid MLP classifier
        # Graph features (hidden_channels * 4) + Demographic features (demo_dim)
        input_dim = hidden_channels * 4 + demo_dim
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with attributes:
                - x: Node features [N, C_in] or [B, N, C_in]
                - edge_index: Edge indices [2, E]
                - edge_weight: Edge weights [E] (optional)
                - batch: Batch vector [N] (for batched graphs)
        Returns:
            logits: Class logits [B, num_classes]
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Graph Conv Layer 1
        x = self.conv1(x, edge_index, edge_weight)
        
        # Graph Conv Layer 2
        x = self.conv2(x, edge_index, edge_weight)
        
        # TopK Pooling
        x, edge_index, edge_weight, batch, _, _ = self.pool(x, edge_index, edge_weight, batch)
        
        # Graph Conv Layer 3
        x = self.conv3(x, edge_index, edge_weight)
        
        # Global pooling: concatenate mean and max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_emb = torch.cat([x_mean, x_max], dim=1)  # [B, hidden_channels * 4]
        
        # Hybrid strategy: concatenate demographic features
        if self.demo_dim > 0 and hasattr(data, 'demo'):
            demo = data.demo.view(-1, self.demo_dim)
            x = torch.cat([graph_emb, demo], dim=1)
        else:
            x = graph_emb
        
        # MLP
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x
    
    def get_embeddings(self, data):
        """
        Extract graph-level embeddings before final classification.
        
        Args:
            data: PyG Data object
        Returns:
            embeddings: Graph-level embeddings [B, hidden_channels * 4]
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = self.conv1(x, edge_index, edge_weight)
        x = self.conv2(x, edge_index, edge_weight)
        x, edge_index, edge_weight, batch, _, _ = self.pool(x, edge_index, edge_weight, batch)
        x = self.conv3(x, edge_index, edge_weight)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        return x


class BrainGNNSimplified(nn.Module):
    """
    Simplified BrainGNN with fewer parameters.
    Useful when dealing with small datasets.
    """
    def __init__(self, in_channels=116, hidden_channels=32, num_classes=2, 
                 pooling_ratio=0.5, dropout_rate=0.5, conv_type='gcn', demo_dim=0):
        super(BrainGNNSimplified, self).__init__()
        
        self.demo_dim = demo_dim
        
        # Two graph conv layers with BatchNorm
        if conv_type == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        elif conv_type == 'graph':
            self.conv1 = GraphConv(in_channels, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, hidden_channels * 2)
        else:
            raise ValueError(f"conv_type must be 'gcn' or 'graph', got {conv_type}")
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        
        # Hybrid Classifier
        input_dim = hidden_channels * 4 + demo_dim
        self.fc1 = nn.Linear(input_dim, hidden_channels)
        self.bn_fc = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Conv 1
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv 2
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Global Readout
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_emb = torch.cat([x_mean, x_max], dim=1)
        
        # Hybrid
        if self.demo_dim > 0 and hasattr(data, 'demo'):
            demo = data.demo.view(-1, self.demo_dim)
            x = torch.cat([graph_emb, demo], dim=1)
        else:
            x = graph_emb
        
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    if not PYG_AVAILABLE:
        print("Cannot test BrainGNN without PyTorch Geometric")
        exit(0)
    
    # Test the model
    n_nodes = 116
    batch_size = 4
    
    # Create dummy data
    x = torch.randn(n_nodes, n_nodes)  # Node features = FC matrix rows
    
    # Create fully connected edge index (without self-loops)
    edge_index = []
    edge_weight = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge_index.append([i, j])
                edge_weight.append(np.random.randn())
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    
    # Create model
    model = BrainGNN(in_channels=n_nodes, hidden_channels=64, num_classes=2)
    
    # Forward pass
    output = model(data)
    print(f"Single sample output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test batched data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    dataset = [data.clone() for _ in range(batch_size)]
    loader = PyGDataLoader(dataset, batch_size=batch_size)
    
    batch = next(iter(loader))
    output_batch = model(batch)
    print(f"\nBatch output shape: {output_batch.shape}")
    print(f"Expected batch size: {batch_size}")
