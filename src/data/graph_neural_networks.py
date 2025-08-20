import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class SupplyChainGNN(nn.Module):
    def __init__(self, num_features=6, hidden_dim=64, num_classes=2):
        super(SupplyChainGNN, self).__init__()
        self.gat1 = GATConv(num_features, hidden_dim, heads=4, dropout=0.3)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gat2(x, edge_index))

        if batch is not None:
            x = global_mean_pool(x, batch)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def prepare_graph_data(company_df, relationships_df, time_series_df):
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

    # Get latest risk scores for each company
    latest_risks = time_series_df.groupby('company_id')['risk_score'].mean().reset_index()
    latest_disruptions = time_series_df.groupby('company_id')['disruption_occurred'].max().reset_index()

    for _, company in company_df_copy.iterrows():
        company_id = company['company_id']
        risk_score = latest_risks[latest_risks['company_id'] == company_id]['risk_score'].iloc[0]

        features = [
            company['region_encoded'],
            company['industry_encoded'],
            company['size_encoded'],
            company['financial_health'],
            risk_score,
            len(relationships_df[relationships_df['supplier_id'] == company_id])  # out-degree
        ]

        node_features.append(features)
        disruption = latest_disruptions[latest_disruptions['company_id'] == company_id]['disruption_occurred'].iloc[0]
        risk_labels.append(disruption)

    # Prepare edge indices
    edge_index = []
    for _, rel in relationships_df.iterrows():
        edge_index.append([rel['supplier_id'], rel['buyer_id']])

    # Convert to tensors
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    y = torch.LongTensor(risk_labels)

    return Data(x=x, edge_index=edge_index, y=y)

def train_gnn_model(graph_data):
    # Create model
    model = SupplyChainGNN(num_features=graph_data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()

    # Create train/test masks
    num_nodes = graph_data.x.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_indices = torch.randperm(num_nodes)[:int(0.8 * num_nodes)]
    test_indices = torch.randperm(num_nodes)[int(0.8 * num_nodes):]

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    # Training loop
    model.train()
    for epoch in range(100):
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

    return model
