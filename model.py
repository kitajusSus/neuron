
# model.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import joblib

class MATIMOSIEJ(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super(MATIMOSIEJ, self).__init__()
        self.mAtI = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.mOsIeJ = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  #dane musza byc znormalizowane czyli w przedziale [0,1]
        )

    def forward(self, x):
        encoded = self.mAtI(x)
        decoded = self.mOsIeJ(encoded)
        return decoded, encoded

def load_data(file_path):
    data = pd.read_csv(file_path, sep=';', header=None)
    
    def clean_and_convert(value):
        if isinstance(value, str):
            value = value.strip()
            if value in ['', ' ']:
                return 0.0
            try:
                if 'lis' in value.lower():
                    value = value.lower().replace('lis.', '').replace('lis', '')
                return float(value)
            except ValueError:
                return 0.0
        return value

    for column in data.columns:
        data[column] = data[column].map(clean_and_convert)
    
    return data.values

def group_data(data, group_size=200):
    n_samples = data.shape[0]
    n_groups = n_samples // group_size
    data = data[:n_groups * group_size]
    grouped_data = data.reshape(n_groups, group_size, -1)
    group_features = np.mean(grouped_data, axis=1)
    return group_features

def train_autoencoder(data_tensor, input_dim, hidden_dim=128, bottleneck_dim=10, epochs=100, batch_size=32):
    model = MATIMOSIEJ(input_dim, hidden_dim, bottleneck_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    
    return model

def train_event_classifier(model, data_tensor, n_clusters=4):
    model.eval()
    with torch.no_grad():
        _, features = model(data_tensor)
        features = features.numpy()
    kmeans = KMeans(n_clusters=n_clusters)
    event_labels = kmeans.fit_predict(features)
    return event_labels, features, kmeans

def save_models(autoencoder, kmeans, autoencoder_path='BOMBACLAT_model.pth', kmeans_path='kmeans_model.pkl'):
    torch.save(autoencoder.state_dict(), autoencoder_path)
    joblib.dump(kmeans, kmeans_path)

def load_models(autoencoder, kmeans, autoencoder_path='BOMBACLAT_model.pth', kmeans_path='kmeans_model.pkl'):
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    autoencoder.eval()
    kmeans = joblib.load(kmeans_path)
    return autoencoder, kmeans

