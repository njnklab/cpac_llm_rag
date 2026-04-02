"""
BrainNetCNN Model Implementation
================================
Based on the BrainNetCNN paper for brain network connectivity analysis.
Implements E2E (Edge-to-Edge), E2N (Edge-to-Node), and N2G (Node-to-Graph) layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class E2EBlock(nn.Module):
    """
    Edge-to-Edge Block
    Learns patterns between connections by aggregating row and column information.
    
    For a connectivity matrix A, this block:
    1. Aggregates information along rows
    2. Aggregates information along columns  
    3. Combines them to learn edge-level patterns
    """
    def __init__(self, in_channels, out_channels, n_nodes=116, bias=True):
        super(E2EBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_nodes = n_nodes
        
        # Row-wise convolution (1xN kernel applied to each row)
        self.row_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, n_nodes), bias=bias)
        
        # Column-wise convolution (Nx1 kernel applied to each column)
        self.col_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(n_nodes, 1), bias=bias)
        
    def forward(self, x):
        """
        Args:
            x: [B, C_in, N, N] - batch of connectivity matrices
        Returns:
            [B, C_out, N, N] - transformed connectivity matrices
        """
        # Apply row-wise convolution: treats each row as a sequence
        row_out = self.row_conv(x)  # [B, C_out, N, 1]
        row_out = row_out.expand(-1, -1, -1, self.n_nodes)  # [B, C_out, N, N]
        
        # Apply column-wise convolution: treats each column as a sequence
        col_out = self.col_conv(x)  # [B, C_out, 1, N]
        col_out = col_out.expand(-1, -1, self.n_nodes, -1)  # [B, C_out, N, N]
        
        # Combine row and column information
        out = row_out + col_out
        return out


class E2NBlock(nn.Module):
    """
    Edge-to-Node Block
    Compresses edge-level features to node-level representations.
    
    Transforms from edge space to node space by aggregating 
    all edges connected to each node.
    """
    def __init__(self, in_channels, out_channels, n_nodes=116, bias=True):
        super(E2NBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_nodes = n_nodes
        
        # Convolve across rows to aggregate edges for each node
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, n_nodes), bias=bias)
        
    def forward(self, x):
        """
        Args:
            x: [B, C_in, N, N] - edge-level features
        Returns:
            [B, C_out, N, 1] - node-level features
        """
        out = self.conv(x)  # [B, C_out, N, 1]
        return out


class N2GBlock(nn.Module):
    """
    Node-to-Graph Block
    Aggregates node-level features to graph-level representation.
    
    Transforms from node space to graph space, producing a single
    vector representing the entire brain network.
    """
    def __init__(self, in_channels, out_channels, n_nodes=116, bias=True):
        super(N2GBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_nodes = n_nodes
        
        # Aggregate all nodes into graph representation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(n_nodes, 1), bias=bias)
        
    def forward(self, x):
        """
        Args:
            x: [B, C_in, N, 1] - node-level features
        Returns:
            [B, C_out, 1, 1] - graph-level representation
        """
        out = self.conv(x)  # [B, C_out, 1, 1]
        return out


class BrainNetCNN(nn.Module):
    """
    BrainNetCNN Model
    
    Args:
        n_nodes: Number of ROIs (default 116 for Yeo7 atlas)
        n_classes: Number of output classes (default 2)
        dropout_rate: Dropout probability
        demo_dim: Dimension of demographic features (0 if none)
    """
    def __init__(self, n_nodes=116, n_classes=2, dropout_rate=0.5, demo_dim=0):
        super(BrainNetCNN, self).__init__()
        
        self.n_nodes = n_nodes
        self.n_classes = n_classes
        self.demo_dim = demo_dim
        
        # E2E blocks: learn edge-level connectivity patterns
        self.e2e_1 = E2EBlock(1, 32, n_nodes=n_nodes)
        self.bn_e2e_1 = nn.BatchNorm2d(32)
        self.dropout_e2e_1 = nn.Dropout2d(dropout_rate / 2)
        
        self.e2e_2 = E2EBlock(32, 64, n_nodes=n_nodes)
        self.bn_e2e_2 = nn.BatchNorm2d(64)
        self.dropout_e2e_2 = nn.Dropout2d(dropout_rate / 2)
        
        # E2N block: compress to node level
        self.e2n = E2NBlock(64, 128, n_nodes=n_nodes)
        self.bn_e2n = nn.BatchNorm2d(128)
        self.dropout_e2n = nn.Dropout2d(dropout_rate)
        
        # N2G block: compress to graph level
        self.n2g = N2GBlock(128, 256, n_nodes=n_nodes)
        self.bn_n2g = nn.BatchNorm2d(256)
        
        # Hybrid Fully connected classifier
        # Brain features (256) + Demographic features (demo_dim)
        self.input_dim = 256 + demo_dim
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout_fc = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, n_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, demo=None):
        """
        Args:
            x: [B, 1, N, N] - batch of FC matrices
            demo: [B, demo_dim] - optional demographic features
        """
        # Feature extraction
        x = self.e2e_1(x)
        x = self.bn_e2e_1(x)
        x = F.relu(x)
        x = self.dropout_e2e_1(x)
        
        x = self.e2e_2(x)
        x = self.bn_e2e_2(x)
        x = F.relu(x)
        x = self.dropout_e2e_2(x)
        
        x = self.e2n(x)
        x = self.bn_e2n(x)
        x = F.relu(x)
        x = self.dropout_e2n(x)
        
        x = self.n2g(x)
        x = self.bn_n2g(x)
        x = F.relu(x)
        
        brain_emb = x.view(x.size(0), -1)  # [B, 256]
        
        # Hybrid: combine with demographics
        if demo is not None and self.demo_dim > 0:
            x = torch.cat([brain_emb, demo], dim=1)
        else:
            x = brain_emb
        
        # FC layers
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        x = self.fc2(x)
        
        return x


class BrainNetCNNSimplified(nn.Module):
    """
    Simplified BrainNetCNN with Demographic support.
    """
    def __init__(self, n_nodes=116, n_classes=2, dropout_rate=0.5, demo_dim=0):
        super(BrainNetCNNSimplified, self).__init__()
        
        self.n_nodes = n_nodes
        self.demo_dim = demo_dim
        
        # Smaller architecture
        self.e2e_1 = E2EBlock(1, 16, n_nodes=n_nodes)
        self.bn_e2e_1 = nn.BatchNorm2d(16)
        
        self.e2e_2 = E2EBlock(16, 32, n_nodes=n_nodes)
        self.bn_e2e_2 = nn.BatchNorm2d(32)
        
        self.e2n = E2NBlock(32, 64, n_nodes=n_nodes)
        self.bn_e2n = nn.BatchNorm2d(64)
        
        self.n2g = N2GBlock(64, 128, n_nodes=n_nodes)
        self.bn_n2g = nn.BatchNorm2d(128)
        
        # Hybrid Classifier
        self.fc1 = nn.Linear(128 + demo_dim, 32)
        self.bn_fc = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(32, n_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, demo=None):
        x = F.relu(self.bn_e2e_1(self.e2e_1(x)))
        x = F.relu(self.bn_e2e_2(self.e2e_2(x)))
        x = F.relu(self.bn_e2n(self.e2n(x)))
        x = F.relu(self.bn_n2g(self.n2g(x)))
        brain_emb = x.view(x.size(0), -1)  # [B, 128]
        
        if demo is not None and self.demo_dim > 0:
            x = torch.cat([brain_emb, demo], dim=1)
        else:
            x = brain_emb
            
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



if __name__ == "__main__":
    # Test the model
    batch_size = 4
    n_nodes = 116
    
    model = BrainNetCNN(n_nodes=n_nodes, n_classes=2)
    
    # Test input
    x = torch.randn(batch_size, 1, n_nodes, n_nodes)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test simplified version
    model_simple = BrainNetCNNSimplified(n_nodes=n_nodes, n_classes=2)
    output_simple = model_simple(x)
    print(f"\nSimplified model output shape: {output_simple.shape}")
    print(f"Simplified model parameters: {sum(p.numel() for p in model_simple.parameters()):,}")
