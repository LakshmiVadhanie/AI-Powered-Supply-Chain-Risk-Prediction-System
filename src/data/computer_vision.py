"""
Computer Vision model for satellite image analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np


class ActivityDetectionCNN(nn.Module):
    """CNN model for detecting factory activity from satellite images."""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 1):
        super(ActivityDetectionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate size after convolutions: 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.pool(F.relu(self.conv1(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv2(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv3(x)))  # 16 -> 8
        
        x = x.view(-1, 128 * 8 * 8)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        
        return x


def prepare_image_data(satellite_images: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare image data for training."""
    X_images = []
    y_activity = []
    
    for img_data in satellite_images:
        img_array = img_data['image_array']
        # Normalize to [0, 1] and change to CHW format
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1) / 255.0
        X_images.append(img_tensor)
        y_activity.append(img_data['activity_level'])
    
    X_images = torch.stack(X_images)
    y_activity = torch.FloatTensor(y_activity).unsqueeze(1)
    
    return X_images, y_activity


def train_cv_model(
    satellite_images: List[Dict[str, Any]], 
    epochs: int = 20,
    learning_rate: float = 0.001,
    train_split: float = 0.8,
    device: str = 'cpu'
) -> ActivityDetectionCNN:
    """Train the computer vision model."""
    
    # Prepare data
    X_images, y_activity = prepare_image_data(satellite_images)
    
    # Split data
    n_train = int(train_split * len(X_images))
    X_train, X_test = X_images[:n_train], X_images[n_train:]
    y_train, y_test = y_activity[:n_train], y_activity[n_train:]
    
    # Move to device
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)
    
    # Initialize model
    model = ActivityDetectionCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f'CV Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f'CV Test Loss: {test_loss.item():.4f}')
    
    return model


def predict_activity_levels(
    model: ActivityDetectionCNN, 
    satellite_images: List[Dict[str, Any]],
    device: str = 'cpu'
) -> List[float]:
    """Predict activity levels for satellite images."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for img_data in satellite_images:
            img_array = img_data['image_array']
            img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            
            prediction = model(img_tensor).cpu().item()
            predictions.append(prediction)
    
    return predictions
