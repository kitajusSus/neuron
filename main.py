# main.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import joblib

from model import (
    load_data, 
    group_data, 
    train_autoencoder, 
    train_event_classifier, 
    save_models, 
    load_models, 
    MATIMOSIEJ
)
from neuron_eeg import (
    BOMBACLAT, 
    load_data_from_folder, 
    EEGDataset, 
    train_model, 
    detect_anomalies
)
from rysowanie_wyników import plot_waveform_with_classifications, plot_reconstruction

def main():
    print("Witaj w programie EEG Analyzer!")
    print("Wybierz opcję:")
    print("1. Trenowanie modelu na nowych danych")
    print("2. Użycie istniejącego modelu do analizy nowych danych")
    choice = input("Twój wybór (1/2): ").strip()

    if choice == '1':
        train_model_flow()
    elif choice == '2':
        use_model_flow()
    else:
        print("Nieprawidłowy wybór. Zakończenie programu.")

def train_model_flow():
    # Parametry
    file_path = input("Podaj nazwę pliku z danymi do treningu (np. dane1.csv): ").strip()
    group_size = int(input("Podaj rozmiar grupy (np. 431): ").strip())
    n_clusters = int(input("Podaj liczbę klastrów (np. 4): ").strip())
    epochs = int(input("Podaj liczbę epok treningu (np. 100): ").strip())
    batch_size = int(input("Podaj rozmiar batcha (np. 32): ").strip())

    # Ładowanie i przygotowanie danych
    data = load_data(file_path)
    input_dim = data.shape[1]
    data_grouped = group_data(data, group_size)

    # Normalizacja
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_grouped)
    data_tensor = torch.from_numpy(data_scaled).float()

    # Trening autoenkodera
    print("Trening autoenkodera...")
    autoencoder = train_autoencoder(
        data_tensor, 
        input_dim=input_dim, 
        hidden_dim=128, 
        bottleneck_dim=10, 
        epochs=epochs, 
        batch_size=batch_size
    )

    # Klasyfikacja zdarzeń
    print("Klasyfikacja zdarzeń...")
    event_labels, features, kmeans = train_event_classifier(autoencoder, data_tensor, n_clusters=n_clusters)

    # Zapis modeli
    save_models(autoencoder, kmeans)
    print("Modele zapisane jako 'BOMBACLAT_model.pth' i 'kmeans_model.pkl'.")

    # Wizualizacja
    print("Wizualizacja wyników...")
    plt.figure(figsize=(12, 6))
    tsne = TSNE(n_components=1, random_state=0, perplexity=min(30, len(features)-1), n_iter=1000, learning_rate='auto')
    tsne_features = tsne.fit_transform(features)
    plt.scatter(range(len(tsne_features)), tsne_features[:, 0], c=event_labels, cmap='viridis', s=100)
    plt.title(f"Klasyfikacja zdarzeń (grupowane po {group_size} próbek)")
    plt.xlabel("Numer grupy")
    plt.ylabel("Cecha TSNE")
    plt.colorbar(label='Klasa zdarzenia')
    plt.grid(True)
    plt.show()

    # Rysowanie fali z klasyfikacjami
    from rysowanie_wyników import plot_waveform_with_classifications
    plot_waveform_with_classifications(data, event_labels, group_size)

    # Zapis wyników
    save_path = "wyniki_analizy.csv"
    results = pd.DataFrame({
        'Group_Start': range(0, len(event_labels) * group_size, group_size),
        'Group_End': range(group_size, (len(event_labels) + 1) * group_size, group_size),
        'Event_Label': event_labels
    })
    results.to_csv(save_path, index=False)
    print(f"Wyniki zapisane w {save_path}")

def use_model_flow():
    # Ścieżki do modeli
    autoencoder_path = "BOMBACLAT_model.pth"
    kmeans_path = "kmeans_model.pkl"

    if not os.path.exists(autoencoder_path) or not os.path.exists(kmeans_path):
        print("Modele nie zostały znalezione. Najpierw przetrenuj model.")
        return

    # Wczytanie nazwy pliku od użytkownika
    file_path = input("Podaj nazwę pliku z danymi do analizy (np. dane2.csv): ").strip()
    group_size = int(input("Podaj rozmiar grupy (np. 431): ").strip())

    # Ładowanie danych
    print("Ładowanie danych...")
    data = load_data(file_path)
    input_dim = data.shape[1]
    data_grouped = group_data(data, group_size)

    # Normalizacja
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_grouped)
    data_tensor = torch.from_numpy(data_scaled).float()

    # Inicjalizacja modelu
    autoencoder = BOMBACLAT(input_dim, hidden_dim=128, bottleneck_dim=10)
    from model import load_models
    autoencoder, kmeans = load_models(autoencoder, None, autoencoder_path, kmeans_path)

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
    tsne = TSNE(n_components=1, random_state=0, perplexity=min(30, len(features)-1), n_iter=1000, learning_rate='auto')
    tsne_features = tsne.fit_transform(features)
    plt.scatter(range(len(tsne_features)), tsne_features[:, 0], c=event_labels, cmap='viridis', s=100)
    plt.title(f"Klasyfikacja zdarzeń (grupowane po {group_size} próbek)")
    plt.xlabel("Numer grupy")
    plt.ylabel("Cecha TSNE")
    plt.colorbar(label='Klasa zdarzenia')
    plt.grid(True)
    plt.show()

    # Rysowanie fali z klasyfikacjami
    from rysowanie_wyników import plot_waveform_with_classifications
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

if __name__ == "__main__":
    main()
