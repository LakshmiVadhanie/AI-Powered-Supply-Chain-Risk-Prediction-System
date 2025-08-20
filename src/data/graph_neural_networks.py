"""
Graph Neural Network model for supply chain relationship analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


class SupplyChainGNN(nn.Module):
    """Graph Attention Network for supply chain risk prediction."""
    
    def __init__(
        self, 
        num_features: int = 6, 
        hidden_dim: int = 64, 
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super(SupplyChainGNN, self).__init__()
        
        self.gat1 = GATConv(
            num_features, 
            hidden_dim, 
            heads=num_heads, 
            dropout=dropout
        )
        self.gat2 = GATConv(
            hidden_dim * num_heads, 
            hidden_dim, 
            heads=1, 
            dropout=dropout
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None) -> torch.Tensor:
        """Forward pass through the GNN."""
        # First GAT layer
        x = F.relu(self.gat1(x, edge_index))
        x = self.dropout(x)
        
        # Second GAT layer
        x = F.relu(self.gat2(x, edge_index))
        
        # Global pooling if batch is provided (for graph-level predictions)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def prepare_graph_data(
    company_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    time_series_df: pd.DataFrame
) -> Data:
    """Prepare graph data for GNN training."""
    
    # Prepare node features
    node_features = []
    risk_labels = []
    
    # Encode categorical features
    region_encoder = LabelEncoder()
    industry_encoder = LabelEncoder()
    size_encoder = LabelEncoder()
    
    company_df_copy = company_df.copy()
    company_df_copy['region_encoded'] = region_encoder.fit_transform(company_df_copy['region'])
    company_df_copy['industry_encoded'] = industry_encoder.fit_transform(company_df_copy['industry'])
    company_df_copy['size_encoded'] = size_encoder.fit_transform(company_df_copy['size'])
    
    # Get aggregated risk scores for each company
    latest_risks = time_series_df.groupby('company_id')['risk_score'].mean().reset_index()
    latest_disruptions = time_series_df.groupby('company_id')['disruption_occurred'].max().reset_index()
    
    # Build node features
    for _, company in company_df_copy.iterrows():
        company_id = company['company_id']
        
        # Get risk score for this company
        risk_row = latest_risks[latest_risks['company_id'] == company_id]
        risk_score = risk_row['risk_score'].iloc[0] if not risk_row.empty else 0.5
        
        # Count outgoing edges (supplier relationships)
        out_degree = len(relationships_df[relationships_df['supplier_id'] == company_id])
        
        features = [
            company['region_encoded'],
            company['industry_encoded'],
            company['size_encoded'],
            company['financial_health'],
            risk_score,
            out_degree
        ]
        
        node_features.append(features)
        
        # Get disruption label
        disruption_row = latest_disruptions[latest_disruptions['company_id'] == company_id]
        disruption = disruption_row['disruption_occurred'].iloc[0] if not disruption_row.empty else 0
        risk_labels.append(disruption)
    
    # Prepare edge indices
    edge_index = []
    for _, rel in relationships_df.iterrows():
        edge_index.append([rel['supplier_id'], rel['buyer_id']])
    
    # Convert to tensors
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edge_index).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    y = torch.LongTensor(risk_labels)
    
    return Data(x=x, edge_index=edge_index, y=y)


def train_gnn_model(
    company_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    time_series_df: pd.DataFrame,
    epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    train_split: float = 0.8,
    device: str = 'cpu'
) -> SupplyChainGNN:
    """Train the Graph Neural Network model."""
    
    # Prepare graph data
    graph_data = prepare_graph_data(company_df, relationships_df, time_series_df)
    graph_data = graph_data.to(device)
    
    # Create model
    model = SupplyChainGNN(num_features=graph_data.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()
    
    # Create train/test masks
    num_nodes = graph_data.x.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Random split
    indices = torch.randperm(num_nodes)
    train_size = int(train_split * num_nodes)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out[train_mask], graph_data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'GNN Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(graph_data.x, graph_data.edge_index)
        test_correct = pred[test_mask].max(1)[1].eq(graph_data.y[test_mask]).sum().item()
        test_acc = test_correct / test_mask.sum().item()
        print(f'GNN Test Accuracy: {test_acc:.4f}')
    
    return model, graph_data


def predict_graph_risks(
    model: SupplyChainGNN,
    graph_data: Data,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate risk predictions using the trained GNN."""
    
    model.eval()
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        pred = model(graph_data.x, graph_data.edge_index)
        # Return probability of disruption (class 1)
        probs = F.softmax(pred, dim=1)[:, 1]
    
    return probs
