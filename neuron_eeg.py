# neuron_eeg.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label

class EEGDataset:
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class BOMBACLAT(nn.Module):
    def __init__(self):
        super(BOMBACLAT, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_data_from_folder(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    data_list = []
    for file in files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path, header=None, names=['C4'])
        df['C4'] = pd.to_numeric(df['C4'], errors='coerce').fillna(df['C4'].mean())
        data = df['C4'].values.astype(np.float32).reshape(-1, 1)
        data_list.append(data)
    return np.concatenate(data_list, axis=0) if data_list else np.array([])

def train_model(model, dataloader, criterion, optimizer, epochs=100):
    model.train()
    training_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        training_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    return training_losses

def detect_anomalies(model, data, threshold_percentile=95, window_size=25):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)
        reconstructed = model(data_tensor).numpy()
        loss = np.mean((data.reshape(-1, 1) - reconstructed) ** 2, axis=1)

    threshold = np.percentile(loss, threshold_percentile)
    anomalies = loss > threshold

    # Use binary dilation to connect nearby anomalies
    structure = np.ones(window_size)
    dilated_anomalies = binary_dilation(anomalies, structure=structure)

    # Label connected regions
    labeled_regions, num_regions = label(dilated_anomalies)

    return reconstructed, loss, dilated_anomalies, labeled_regions, num_regions, threshold
