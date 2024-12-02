import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns  # Add this import at the top with other imports

# Ładowanie danych z pliku CSV
def load_data(file_path):
    # Read CSV with semicolon separator
    data = pd.read_csv(file_path, sep=';', header=None)
    
    # Clean data
    def clean_and_convert(value):
        if isinstance(value, str):
            # Remove whitespace
            value = value.strip()
            # Handle empty strings
            if value == '' or value == ' ':
                return 0.0
            # Handle 'lis.75' and similar cases
            try:
                if 'lis' in value.lower():
                    value = value.lower().replace('lis.', '').replace('lis', '')
                return float(value)
            except ValueError:
                return 0.0  # Return 0 for any value we can't convert
        return value

    # Apply cleaning to all columns using map
    for column in data.columns:
        data[column] = data[column].map(clean_and_convert)
    
    return data.values

# BOMBACLAT Autoencoder  KLASYCZNY Z DEFINICJI ZERO UDZIWNIEŃ
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        self.mAtI = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.mOsiEj = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Opcjonalnie, w zależności od danych
        )

    def forward(self, x):
        encoded = self.mAtI(x)
        decoded = self.mOsiEj(encoded)
        return decoded, encoded

# Funkcja do grupowania danych
def group_data(data, group_size=200):
    """robimy dobre rozmiary == takie same dla danych by wszystko było takie same"""
    n_samples = data.shape[0]
    n_groups = n_samples // group_size
    # Trim data to be divisible by group_size
    data = data[:n_groups * group_size]
    # Reshape into groups
    grouped_data = data.reshape(n_groups, group_size, -1)
    # Calculate mean features for each group
    group_features = np.mean(grouped_data, axis=1)
    dwa = int(200)
    ## PĘTLA DO WYSWIETLANIA MOSIEJ 
    for i in range(dwa):
        print("M", "o ", "s", "i", "e", "j")
        time.sleep(1)
        print("W            A                  K           E      U            P")
        time.sleep(1)
    return group_features

# Główny skrypt
if __name__ == "__main__":
    # Parametry
    file_path = "dane2.csv"  # Ścieżka do pliku CSV
    input_dim = None  # gdy juz załadujemy dane to zmieni sie na data.shape[1]
    group_size = 431 # Rozmiar grupy dla jednego zdarzenia zmienić na liczbę np.431
    
    # Ładowanie danych
    data = load_data(file_path)
    input_dim = data.shape[1]  # WYMIAR DANYCH NAJLEPIEJ 1 I MAMY WSZYSTKO wywalone
    
    hidden_dim = 128
    bottleneck_dim = 10 # Liczba cech do wyodrębnienia
    epochs = 100
    batch_size = 32
    n_clusters = 4 # Liczba oczekiwanych zdarzeń do wykrycia

    # Grupowanie danych
    data_grouped = group_data(data, group_size)

    # Normalizacja danych (jeśli nie zostało to już wykonane)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_grouped)

    # Konwersja danych do tensorów PyTorch
    data_tensor = torch.from_numpy(data_scaled).float()

    # Funkcja do trenowania klasyfikatora zdarzeń
    def train_event_classifier(data_tensor, input_dim, hidden_dim, bottleneck_dim, epochs, batch_size, n_clusters):
        # Tworzenie modelu, kryterium i optymizatora
        model = Autoencoder(input_dim, hidden_dim, bottleneck_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Trening modelu
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i:i+batch_size]
                if len(batch) < batch_size:
                    continue

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs, _ = model(batch)
                loss = criterion(outputs, batch)

                # Backward pass i aktualizacja
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Print average loss for the epoch
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.6f}')

        # Użycie modelu do wyodrębnienia cech
        with torch.no_grad():
            _, features = model(data_tensor)
            features = features.numpy()

        # Klasyfikacja KMeans na cechach
        kmeans = KMeans(n_clusters=n_clusters)
        event_labels = kmeans.fit_predict(features)

        return event_labels, features

    # Trenowanie modelu klasyfikacji zdarzeń
    event_labels, features = train_event_classifier(data_tensor, input_dim, hidden_dim, bottleneck_dim, epochs, batch_size, n_clusters)

    # Wizualizacja klasyfikacji (t-SNE)
    plt.figure(figsize=(12, 6))
    
    # Calculate appropriate perplexity (should be smaller than n_samples)
    n_samples = len(features)
    perplexity = min(30, n_samples - 1)  # default is 30, but needs to be < n_samples
    
    tsne = TSNE(
        n_components=1,
        random_state=0,
        perplexity=perplexity,
        n_iter=1000,
        learning_rate='auto'
    )
    
    tsne_features = tsne.fit_transform(features)
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(tsne_features)), tsne_features[:, 0], c=event_labels, cmap='viridis', s=100)
    plt.title(f"Klasyfikacja zdarzeń w danych EEG (grupowane po {group_size} próbek)")
    plt.xlabel("Numer grupy")
    plt.ylabel("Cecha TSNE")
    plt.colorbar(label='Klasa zdarzenia')
    plt.grid(True)
    
    # Wizualizacja pełnej fali z klasyfikacjami
    plot_waveform_with_classifications(data, event_labels, group_size)

"""
    # Zapisywanie wyników do pliku (opcjonalnie)
    results = pd.DataFrame({
        'Group_Start': range(0, len(event_labels) * group_size, group_size),
        'Group_End': range(group_size, (len(event_labels) + 1) * group_size, group_size),
        'Event_Label': event_labels,
        'Feature_1': features[:, 0],
        #'Feature_2': features[:, 1],  # Możesz dodać więcej cech, jeśli chcesz
        #... (dla pozostałych cech, dopisać tak samo, nie uzyywac jak ma sie bardzo duzo danych bo moze zuzyc duzo mocy obliczeniowej i pamieci RAM ALE NIE SPRAWDZAŁEM)
    })
    results.to_csv("wyniki_klasyfikacji.csv", index=False)
"""
