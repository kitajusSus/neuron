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
from rysowanie_wyników import plot_waveform_with_classifications
import model



if __name__ == "__main__":
    # Ścieżki do modeli
    autoencoder_path = "model_autoencoder.pth"
    kmeans_path = "kmeans_model.pkl"
    
    # Wczytanie nazwy pliku od użytkownika
    file_path = input("Podaj nazwę pliku z danymi (np. dane2.csv): ").strip()
    
    # Parametry
    group_size = 431  # Dostosuj rozmiar grupy do swoich danych
    
    # Ładowanie danych
    print("Ładowanie danych...")
    data = load_data(file_path)
    input_dim = data.shape[1]
    
    # Grupowanie danych
    data_grouped = group_data(data, group_size)
    
    # Normalizacja danych
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_grouped)
    data_tensor = torch.from_numpy(data_scaled).float()
    
    # Wczytanie modeli
    print("Wczytywanie wytrenowanych modeli...")
    autoencoder = Autoencoder(input_dim, hidden_dim=128, bottleneck_dim=10)
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    autoencoder.eval()
    
    kmeans = joblib.load(kmeans_path)
    
    # Wyodrębnianie cech
    print("Ekstrakcja cech...")
    with torch.no_grad():
        _, features = autoencoder(data_tensor)
        features = features.numpy()
    
    # Klasyfikacja
    print("Klasyfikacja zdarzeń...")
    event_labels = kmeans.predict(features)
    
    # Wizualizacja
    print("Wizualizacja wyników...")
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(features)), features[:, 0], c=event_labels, cmap='viridis', s=100)
    plt.title(f"Klasyfikacja zdarzeń (grupowane po {group_size} próbek)")
    plt.xlabel("Numer grupy")
    plt.ylabel("Cechy")
    plt.colorbar(label='Klasa zdarzenia')
    plt.grid(True)
    plt.show()
    
    plot_waveform_with_classifications(data, event_labels, group_size)
    
    # Zapis wyników do pliku
    save_path = "wyniki_analizy.csv"
    print(f"Zapisywanie wyników do pliku {save_path}...")
    results = pd.DataFrame({
        'Group_Start': range(0, len(event_labels) * group_size, group_size),
        'Group_End': range(group_size, (len(event_labels) + 1) * group_size, group_size),
        'Event_Label': event_labels
    })
    results.to_csv(save_path, index=False)
    print("Analiza zakończona!")
