import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return torch.sigmoid(out)

def prepare_time_series_data(time_series_df, sequence_length=30):
    # Group by company and create sequences
    sequences = []
    labels = []

    feature_cols = ['risk_score', 'demand_volatility', 'price_volatility', 'inventory_level']
    scaler = StandardScaler()

    for company_id in time_series_df['company_id'].unique():
        company_data = time_series_df[time_series_df['company_id'] == company_id].sort_values('date')

        if len(company_data) >= sequence_length + 1:
            features = company_data[feature_cols].values
            features_scaled = scaler.fit_transform(features)
            targets = company_data['disruption_occurred'].values

            for i in range(len(features_scaled) - sequence_length):
                sequences.append(features_scaled[i:i+sequence_length])
                labels.append(targets[i+sequence_length])

    return np.array(sequences), np.array(labels), scaler

def train_time_series_model(X_seq, y_seq):
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)

    # Split data
    n_train = int(0.8 * len(X_tensor))
    X_train, X_test = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_test = y_tensor[:n_train], y_tensor[n_train:]

    # Create model
    model = TimeSeriesLSTM(input_size=4)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(30):
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

    return model
