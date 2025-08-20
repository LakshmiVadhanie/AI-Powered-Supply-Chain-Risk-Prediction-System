import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivityDetectionCNN(nn.Module):
    def __init__(self):
        super(ActivityDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

def train_cv_model(satellite_images):
    # Prepare image data
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

    # Split data
    n_train = int(0.8 * len(X_images))
    X_train, X_test = X_images[:n_train], X_images[n_train:]
    y_train, y_test = y_activity[:n_train], y_activity[n_train:]

    # Create model
    model = ActivityDetectionCNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(20):
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
