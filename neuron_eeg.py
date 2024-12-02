import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label
from collections import OrderedDict 

class EEGDataset:
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 1. Definicja klasy modelu BOMBACLAT
class BOMBACLAT(nn.Module):
    def __init__(self):
        super(BOMBACLAT, self).__init__()
        self.mati = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.mosiej = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        encoded = self.mati(x)
        decoded = self.mosiej(encoded)
        return decoded
# Funkcja do ładowania danych. 
def load_data_from_folder(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    data_list = []
    for file in files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path, header=None, names=['C4'])
        df['C4'] = pd.to_numeric(df['C4'], errors='coerce').fillna(df['C4'].mean()) # Zastępowanie brakujących wartości średnią
        data = df['C4'].values.astype(np.float32)
        data = data.reshape(-1, 1)
    return np.concatenate(data_list, axis=0) if data_list else np.array([])


    
# GLOBALNY DOZWOLONY OBIEKT
torch.serialization.add_safe_globals([BOMBACLAT])

# Wykrywanie anomalii w sygnałach EEG

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

def plot_results(df, reconstructed, loss, dilated_anomalies, labeled_regions, num_regions, threshold):
    # Oryginalne dane i odtworzone dane: 
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['C4'], label='Original Data', color='blue', alpha=0.6)
    plt.plot(df.index, reconstructed, label='Reconstructed Data', color='green', alpha=0.6)

    # Anomaly Regions Plot
    colors = plt.cm.rainbow(np.linspace(0, 1, num_regions+1))
    y_min, y_max = df['C4'].min(), df['C4'].max()

    for i in range(1, num_regions+1):
        region_mask = labeled_regions == i
        if region_mask.any():
            plt.fill_between(df.index[region_mask], 
                            [y_min] * sum(region_mask), 
                            [y_max] * sum(region_mask),
                            color=colors[i], alpha=0.3,
                            label=f'Anomaly Region {i}')

    plt.title('EEG Signal Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Reconstruction Error Plot
    plt.figure(figsize=(15, 4))
    plt.plot(df.index, loss, label='Reconstruction Error', color='purple')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

    loss_max = loss.max()
    for i in range(2, num_regions+1):
        region_mask = labeled_regions == i
        if region_mask.any():
            plt.fill_between(df.index[region_mask], 
                            [0] * sum(region_mask), 
                            [loss_max] * sum(region_mask),
                            color=colors[i], alpha=0.3)

    plt.title('Reconstruction Error with Anomaly Regions')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # Load and prepare data
    good_data = load_data_from_folder('good')
    bad_data = load_data_from_folder('bad')

    # Prepare dataset
    combined_data = np.concatenate((good_data, bad_data))
    dataset = EEGDataset(combined_data)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model and training setup
    model_path = 'BOMBACLAT_model.pth'
    model = BOMBACLAT()
    
    if os.path.exists(model_path):
        model = torch.load(model_path)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    training_losses = train_model(model, data_loader, criterion, optimizer)

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_losses) + 1), training_losses, 'b-')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # Process test files
    test_files = [f for f in os.listdir('test') if f.endswith('.csv')]
    for test_file in test_files:
        file_path = os.path.join('test', test_file)
        df = pd.read_csv(file_path, header=None, names=['C4'])
        df['C4'] = pd.to_numeric(df['C4'], errors='coerce').fillna(df['C4'].mean())
        data = df['C4'].values.astype(np.float32)

        # Detect anomalies
        reconstructed, loss, dilated_anomalies, labeled_regions, num_regions, threshold = detect_anomalies(model, data)
        
        # Plot results
        plot_results(df, reconstructed, loss, dilated_anomalies, labeled_regions, num_regions, threshold)

        # Print anomaly regions
        for i in range(2, num_regions+1):
            region_mask = labeled_regions == i
            if region_mask.any():
                start_idx = df.index[region_mask].min()
                end_idx = df.index[region_mask].max()
                length = end_idx - start_idx + 2
                print(f'Region {i}: Start={start_idx}, End={end_idx}, Length={length}')

    # Save model
    torch.save(model, 'BOMBACLAT_model.pth')
    print("Model saved successfully.")

if __name__ == '__main__':
    main()
