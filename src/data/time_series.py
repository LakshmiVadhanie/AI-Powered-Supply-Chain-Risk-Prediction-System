"""
Time Series forecasting model for supply chain risk prediction.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


class TimeSeriesLSTM(nn.Module):
    """LSTM model for time series risk prediction."""
    
    def __init__(
        self, 
        input_size: int = 4, 
        hidden_size: int = 64, 
        num_layers: int = 2, 
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(TimeSeriesLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM network."""
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take last time step output
        out = self.fc(out[:, -1, :])
        
        return torch.sigmoid(out)


def prepare_time_series_data(
    time_series_df: pd.DataFrame, 
    sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Prepare time series data for LSTM training."""
    
    sequences = []
    labels = []
    
    feature_cols = ['risk_score', 'demand_volatility', 'price_volatility', 'inventory_level']
    scaler = StandardScaler()
    
    # Process each company separately
    for company_id in time_series_df['company_id'].unique():
        company_data = time_series_df[
            time_series_df['company_id'] == company_id
        ].sort_values('date')
        
        if len(company_data) >= sequence_length + 1:
            # Extract and scale features
            features = company_data[feature_cols].values
            features_scaled = scaler.fit_transform(features)
            targets = company_data['disruption_occurred'].values
            
            # Create sequences
            for i in range(len(features_scaled) - sequence_length):
                sequences.append(features_scaled[i:i+sequence_length])
                labels.append(targets[i+sequence_length])
    
    return np.array(sequences), np.array(labels), scaler


def train_time_series_model(
    time_series_df: pd.DataFrame,
    sequence_length: int = 30,
    epochs: int = 30,
    learning_rate: float = 0.001,
    train_split: float = 0.8,
    device: str = 'cpu'
) -> Tuple[TimeSeriesLSTM, StandardScaler]:
    """Train the time series LSTM model."""
    
    # Prepare data
    X_seq, y_seq, scaler = prepare_time_series_data(time_series_df, sequence_length)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_seq).to(device)
    y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(device)
    
    # Split data
    n_train = int(train_split * len(X_tensor))
    X_train, X_test = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_test = y_tensor[:n_train], y_tensor[n_train:]
    
    # Initialize model
    model = TimeSeriesLSTM(input_size=X_seq.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'TS Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_predictions = (test_outputs > 0.5).float()
        accuracy = (test_predictions == y_test).float().mean()
        print(f'TS Test Accuracy: {accuracy.item():.4f}')
    
    return model, scaler


def predict_time_series_risks(
    model: TimeSeriesLSTM,
    scaler: StandardScaler,
    time_series_df: pd.DataFrame,
    sequence_length: int = 30,
    device: str = 'cpu'
) -> List[float]:
    """Generate time series risk predictions."""
    
    model.eval()
    predictions = []
    
    feature_cols = ['risk_score', 'demand_volatility', 'price_volatility', 'inventory_level']
    
    with torch.no_grad():
        for company_id in time_series_df['company_id'].unique():
            company_data = time_series_df[
                time_series_df['company_id'] == company_id
            ].sort_values('date')
            
            if len(company_data) >= sequence_length:
                # Get last sequence for prediction
                features = company_data[feature_cols].iloc[-sequence_length:].values
                features_scaled = scaler.transform(features)
                
                # Convert to tensor and predict
                sequence_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)
                prediction = model(sequence_tensor).cpu().item()
                predictions.append(prediction)
            else:
                predictions.append(0.5)  # Default prediction
    
    return predictions
