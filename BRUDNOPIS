# BOMBACLAT EEG Wykrywanie Anomalii

## Spis Treści
- [Wstęp](#wstęp)
- [Struktura Projektu](#struktura-projektu)
- [Wymagania](#wymagania)
- [Instalacja](#instalacja)
- [Użycie](#użycie)
  - [Przygotowanie Danych](#przygotowanie-danych)
  - [Uruchomienie Skryptu](#uruchomienie-skryptu)
- [Wyjaśnienie Kodu](#wyjaśnienie-kodu)
  - [Importy i Konfiguracja Środowiska](#importy-i-konfiguracja-środowiska)
  - [Klasa EEGDataset](#klasa-eegdataset)
  - [Klasa Modelu BOMBACLAT](#klasa-modelu-bombaclat)
  - [Funkcja Ładowania Danych](#funkcja-ładowania-danych)
  - [Funkcja Wykrywania Anomalii](#funkcja-wykrywania-anomalii)
  - [Funkcja Wizualizacji](#funkcja-wizualizacji)
  - [Funkcja Główna](#funkcja-główna)
- [Wyniki](#wyniki)
- [Uwagi](#uwagi)
- [Kontakt](#kontakt)

## Wstęp

Witaj w projekcie **BOMBACLAT EEG Wykrywanie Anomalii**! Projekt ten wykorzystuje techniki uczenia maszynowego do wykrywania anomalii w sygnałach EEG (Elektroencefalografia). Przewodnik ten jest przygotowany z myślą o fizykach i innych osobach bez doświadczenia w uczeniu maszynowym, zapewniając szczegółowe wyjaśnienia kodu i zastosowanych metod.

## Struktura Projektu

bombaclat-eeg/ ├── data/ │ ├── good/ │ │ ├── file1.csv │ │ ├── file2.csv │ │ └── ... │ ├── bad/ │ │ ├── file1.csv │ │ ├── file2.csv │ │ └── ... │ └── test/ │ ├── test1.csv │ ├── test2.csv │ └── ... ├── neuron_eeg.py ├── BOMBACLAT_model.pth ├── requirements.txt └── README.md

markdown
o

- **data/**: Zawiera podkatalogi z plikami CSV.
  - **good/**: Pliki CSV reprezentujące normalne dane EEG.
  - **bad/**: Pliki CSV reprezentujące anomalne dane EEG.
  - **test/**: Pliki CSV używane do testowania wykrywania anomalii.
- **neuron_eeg.py**: Główny skrypt Python implementujący wykrywanie anomalii.
- **BOMBACLAT_model.pth**: Zapisany wytrenowany model (generowany po treningu).
- **requirements.txt**: Lista wszystkich zależności Python.
- **README.md**: Ten plik dokumentacji.

## Wymagania

Przed rozpoczęciem upewnij się, że masz zainstalowane poniższe komponenty na swoim systemie:

- **Python**: Wersja 3.7 lub wyższa.
- **pip**: Menedżer pakietów Python.

## Instalacja

1. **Klonowanie Repozytorium**

   Skopiuj repozytorium na swój lokalny komputer:
   
   ```bash
   git clone https://github.com/kitajusSus/neuron.git
   cd neuron
   ```
Tworzenie Środowiska Wirtualnego (Opcjonalnie)

Tworzenie środowiska wirtualnego pomaga zarządzać zależnościami i unikać konfliktów.


```bash
python -m venv venv
```
Instalacja Zależności

Zainstaluj wszystkie wymagane pakiety Python za pomocą requirements.txt:

```bash
pip install -r requirements.txt
```
Jeśli plik requirements.txt nie jest dostępny, możesz zainstalować pakiety ręcznie:

```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn scipy
```
Użycie
Przygotowanie Danych
Organizacja Danych
Format Plików CSV:

Każdy plik CSV powinien zawierać jedną kolumnę nazwaną C4, reprezentującą amplitudy sygnału EEG.
Upewnij się, że w plikach CSV nie ma nagłówków. Jeśli nagłówki istnieją, skrypt je obsłuży.
Separator CSV

Skrypt oczekuje plików CSV z separatorem przecinkowym (,). Upewnij się, że Twoje pliki są w tym formacie.
Uruchomienie Skryptu
Wykonaj główny skrypt, aby wytrenować model i wykryć anomalie:

```bash
python neuron_eeg.py
```
Co Wykonuje Skrypt:

`Ładowanie Danych:` Wczytuje i przetwarza dane EEG z katalogów good i bad.
`Trenowanie Modelu:` Trenuje autoencoder BOMBACLAT na połączonym zbiorze danych.
`Wykrywanie Anomalii:` Używa wytrenowanego modelu do identyfikacji anomalii w danych testowych.
`Wizualizacja:` Generuje wykresy ilustrujące oryginalne vs. odtworzone sygnały oraz zaznacza wykryte anomalie.
`Zapis Modelu:` Zapisuje wytrenowany model do pliku BOMBACLAT_model.pth dla przyszłego użytku.

# Wyjaśnienie Kodu
Poniżej znajduje się szczegółowe wyjaśnienie każdego komponentu w skrypcie neuron_eeg.py.

Importy i Konfiguracja Środowiska
```python
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
```
`os:` Interakcje z systemem operacyjnym, obsługa ścieżek plików i zmiennych środowiskowych.
`pandas (pd):` Manipulacja i analiza danych, szczególnie przydatna do obsługi plików CSV.
`numpy (np):` Obsługa dużych, wielowymiarowych tablic i macierzy danych.
`torch:` Biblioteka PyTorch do budowania i trenowania sieci neuronowych.
`torch.utils.data:` Narzędzia do obsługi ładowania danych.
`torch.nn:` Warstwy sieci neuronowych i funkcje strat.
`torch.optim:` Optymalizatory do trenowania sieci neuronowych.
`seaborn (sns):` Biblioteka do wizualizacji danych oparta na matplotlib.
`matplotlib.pyplot (plt):` Biblioteka do tworzenia statycznych, animowanych i interaktywnych wizualizacji.
`scipy.ndimage:` Funkcje do przetwarzania obrazów wielowymiarowych.
collections.OrderedDict: Podklasa słownika, która pamięta kolejność wstawiania elementów.
Konfiguracja Zmiennej Środowiskowej:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
Ustawienie to pozwala programowi na obsługę potencjalnych problemów z duplikatowymi bibliotekami OpenMP, zapobiegając awariom związanym z ładowaniem bibliotek.
Klasa EEGDataset
python
```
class EEGDataset:
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```
Cel:

EEGDataset to niestandardowa klasa datasetu dostosowana do danych EEG. Przygotowuje dane do ładowania przez sieć neuronową.
Składniki:

__init__: Inicjalizuje dataset, konwertując dane wejściowe (tablicę NumPy) na tensor PyTorch typu float32 i zmieniając jego kształt na (liczba_próbek, 1).

__len__: Zwraca całkowitą liczbę próbek w datasetcie.

__getitem__: Pobiera pojedynczą próbkę z datasetu na podstawie podanego indeksu.

Użycie:

Ta klasa umożliwia PyTorch DataLoader efektywne iterowanie po danych EEG podczas treningu.
Klasa Modelu BOMBACLAT
```python
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
```
**Cel:*

BOMBACLAT to model sieci neuronowej zaprojektowany jako autoencoder. Autoencodery uczą się kompresować (enkodować) dane do niższej przestrzeni wymiarowej, a następnie rekonstruować (dekodować) je z powrotem do oryginalnego wymiaru.

Składniki:

`self.mati (Enkoder):`

- `nn.Linear(1, 32):` Warstwa w pełni połączona transformująca wejście z 1 wymiaru na 32.
- `nn.ReLU():` Funkcja aktywacji wprowadzająca nieliniowość.
Kolejne warstwy stopniowo zmniejszają wymiarowość do 4.

`self.mosiej (Dekoder)`:

`nn.Linear(4, 8)`: Warstwa w pełni połączona zwiększająca wymiarowość z 4 do 8.
`nn.ReLU()`: Funkcja aktywacji.
Kolejne warstwy stopniowo zwiększają wymiarowość z powrotem do 1.
Metoda forward:

Przechodzi wejście x przez enkoder (self.mati), aby uzyskać zakodowane cechy.
Przechodzi zakodowane cechy przez dekoder (self.mosiej), aby odtworzyć oryginalne wejście.
Zwraca odtworzone dane.
Użycie:

Model ten uczy się dokładnie rekonstruować normalne sygnały EEG. Anomalie są wykrywane na podstawie błędów rekonstrukcji.
Funkcja Ładowania Danych
```python
def load_data_from_folder(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    data_list = []
    for file in files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path, header=None, names=['C4'])
        df['C4'] = pd.to_numeric(df['C4'], errors='coerce').fillna(df['C4'].mean()) # Zastępowanie brakujących wartości średnią
        data = df['C4'].values.astype(np.float32)
        data = data.reshape(-1, 1)
        data_list.append(data)
    return np.concatenate(data_list, axis=0) if data_list else np.array([])
```
**Cel:**

Ładuje i przetwarza dane EEG z plików CSV znajdujących się w określonym folderze.
Składniki:

Filtrowanie Plików:

Lista wszystkich plików w podanym folder, które kończą się na .csv.

Czytanie i Czyszczenie Danych:

Wczytuje każdy plik CSV bez nagłówków i przypisuje nazwę kolumny C4.
Konwertuje kolumnę C4 na wartości numeryczne, wymuszając konwersję (errors='coerce'). Wartości, których nie można przekonwertować, stają się NaN.
Zastępuje NaN wartościami średnimi kolumny C4, aby obsłużyć brakujące lub uszkodzone dane.

Formatowanie Danych:

Konwertuje oczyszczone dane C4 na tablicę NumPy typu float32.
Zmienia kształt danych na (liczba_próbek, 1).

Agregacja Danych:

Dodaje każdą przetworzoną tablicę danych do data_list.
Łączy wszystkie tablice w data_list w jedną tablicę NumPy.

Użycie:

Funkcja ta jest używana do ładowania zarówno good (normalnych), jak i bad (anomalnych) danych EEG do trenowania modelu.
Funkcja Wykrywania Anomalii
```python

def detect_anomalies(model, data, threshold_percentile=95, window_size=25):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)
        reconstructed = model(data_tensor).numpy()
        loss = np.mean((data.reshape(-1, 1) - reconstructed) ** 2, axis=1)

    threshold = np.percentile(loss, threshold_percentile)
    anomalies = loss > threshold

    # Użycie dylatacji binarnej do łączenia pobliskich anomalii
    structure = np.ones(window_size)
    dilated_anomalies = binary_dilation(anomalies, structure=structure)

    # Etykietowanie połączonych regionów
    labeled_regions, num_regions = label(dilated_anomalies)

    return reconstructed, loss, dilated_anomalies, labeled_regions, num_regions, threshold
```
Cel:

Identyfikuje anomalie w danych EEG na podstawie błędów rekonstrukcji z modelu autoencoder.
Składniki:

Tryb Ewaluacji Modelu:

Ustawia model w tryb ewaluacji, wyłączając warstwy specyficzne dla treningu, takie jak dropout.
Rekonstrukcja Danych:

Konwertuje dane wejściowe na tensor PyTorch.
Przechodzi dane przez model, uzyskując odtworzone sygnały.
Oblicza Błąd Rekonstrukcji (loss) jako średni błąd kwadratowy (MSE) między oryginalnymi a odtworzonymi sygnałami.

Określenie Tresholdu:

Oblicza próg na podstawie określonego percentyla (threshold_percentile) błędów rekonstrukcji.
Sygnały z błędami powyżej tego progu są uznawane za anomalie.

Udoskonalenie Wykrywania Anomalii:

Stosuje dylatację binarną, aby połączyć pobliskie punkty anomalii, zapewniając, że anomalie obejmujące wiele kolejnych punktów są traktowane jako jedno zdarzenie.
Etykietuje połączone regiony anomalii.
Parametry:

model: Wytrenowany model autoencoder.
data: Dane EEG do analizy.
threshold_percentile: Percentyl do ustawienia progu wykrywania anomalii (domyślnie 95).
window_size: Rozmiar okna dla dylatacji binarnej do łączenia pobliskich anomalii.
# Użycie:

Funkcja Wizualizacji
```python
#kod kolejny
def plot_results(df, reconstructed, loss, dilated_anomalies, labeled_regions, num_regions, threshold):
    # Oryginalne dane i odtworzone dane
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['C4'], label='Oryginalne Dane', color='blue', alpha=0.6)
    plt.plot(df.index, reconstructed, label='Odtworzone Dane', color='green', alpha=0.6)

    # Wizualizacja Regionów Anomalii
    colors = plt.cm.rainbow(np.linspace(0, 1, num_regions+1))
    y_min, y_max = df['C4'].min(), df['C4'].max()

    for i in range(1, num_regions+1):
        region_mask = labeled_regions == i
        if region_mask.any():
            plt.fill_between(df.index[region_mask], 
                            [y_min] * sum(region_mask), 
                            [y_max] * sum(region_mask),
                            color=colors[i], alpha=0.3,
                            label=f'Region Anomalii {i}')

    plt.title('Wykrywanie Anomalii w Sygnałach EEG')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Wykres Błędu Rekonstrukcji
    plt.figure(figsize=(15, 4))
    plt.plot(df.index, loss, label='Błąd Rekonstrukcji', color='purple')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Próg')

    loss_max = loss.max()
    for i in range(2, num_regions+1):
        region_mask = labeled_regions == i
        if region_mask.any():
            plt.fill_between(df.index[region_mask], 
                            [0] * sum(region_mask), 
                            [loss_max] * sum(region_mask),
                            color=colors[i], alpha=0.3)

    plt.title('Błąd Rekonstrukcji z Regionami Anomalii')
    plt.xlabel('Czas')
    plt.ylabel('Błąd')
    plt.grid(True)
    plt.legend()
    plt.show()
```
**Cel:**

Wizualizuje dane EEG, odtworzone sygnały oraz wykryte anomalie w celu łatwej interpretacji wyników.
Składniki:

**Oryginalne vs. Odtworzone Sygnały:**

Rysuje oryginalny sygnał EEG w kolorze niebieskim oraz odtworzony sygnał przez model w kolorze zielonym.
Pomaga zobaczyć, jak dobrze model nauczył się rekonstruować normalne sygnały.

**Regiony Anomalii**:

Podświetla obszary, gdzie wykryto anomalie, używając półprzezroczystych kolorów.

**0Błąd Rekonstrukcji**:

Rysuje błąd rekonstrukcji w kolorze fioletowym.
Dodaje poziomą czerwoną linię przerywaną reprezentującą próg wykrywania anomalii.
Podświetla obszary, gdzie błąd rekonstrukcji przekracza próg, wskazując na anomalie.
Użycie:

Po wykryciu anomalii, użyj tej funkcji do generowania wykresów ilustrujących, gdzie występują anomalie w sygnale EEG.
# Funkcja Główna
```python
def main():
    # Ładowanie i przygotowanie danych
    good_data = load_data_from_folder('data/good')
    bad_data = load_data_from_folder('data/bad')

    # Przygotowanie datasetu
    combined_data = np.concatenate((good_data, bad_data))
    dataset = EEGDataset(combined_data)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Ustawienie modelu i treningu
    model_path = 'BOMBACLAT_model.pth'
    model = BOMBACLAT()
    
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print("Załadowano istniejący model.")
    else:
        print("Trenowanie nowego modelu.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Trenowanie modelu
    training_losses = train_model(model, data_loader, criterion, optimizer)

    # Wizualizacja strat treningowych
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_losses) + 1), training_losses, 'b-')
    plt.title('Strata Treningowa w Przebiegu Epok')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.grid(True)
    plt.show()

    # Przetwarzanie plików testowych
    test_files = [f for f in os.listdir('data/test') if f.endswith('.csv')]
    for test_file in test_files:
        file_path = os.path.join('data/test', test_file)
        df = pd.read_csv(file_path, header=None, names=['C4'])
        df['C4'] = pd.to_numeric(df['C4'], errors='coerce').fillna(df['C4'].mean())
        data = df['C4'].values.astype(np.float32)

        # Wykrywanie anomalii
        reconstructed, loss, dilated_anomalies, labeled_regions, num_regions, threshold = detect_anomalies(model, data)
        
        # Wizualizacja wyników
        plot_results(df, reconstructed, loss, dilated_anomalies, labeled_regions, num_regions, threshold)

        # Wyświetlanie regionów anomalii
        for i in range(2, num_regions+1):
            region_mask = labeled_regions == i
            if region_mask.any():
                start_idx = df.index[region_mask].min()
                end_idx = df.index[region_mask].max()
                length = end_idx - start_idx + 2
                print(f'Region {i}: Start={start_idx}, End={end_idx}, Długość={length}')

    # Zapisanie modelu
    torch.save(model, 'BOMBACLAT_model.pth')
    print("Model został zapisany pomyślnie.")
```
Cel:

Koordynuje cały workflow: ładowanie danych, trenowanie modelu, wykrywanie anomalii, wizualizacja wyników oraz zapis modelu.
Składniki:

Ładowanie Danych:

Wczytuje dane good i bad z ich odpowiednich katalogów.
Łączy dane w jeden zbiór do trenowania modelu.
Przygotowanie Datasetu:

Tworzy instancję EEGDataset z połączonych danych.
Używa DataLoader do obsługi partii i mieszania danych podczas treningu.
Ustawienie Modelu:

Inicjalizuje model BOMBACLAT.
Sprawdza, czy istnieje zapisany model (BOMBACLAT_model.pth). Jeśli tak, ładuje go, aby uniknąć ponownego trenowania.
Konfiguracja Treningu:

Definiuje funkcję straty (MSELoss) odpowiednią dla zadań rekonstrukcji.
Ustawia optymalizator (Adam) do trenowania modelu.
Trenowanie Modelu:

Wywołuje funkcję train_model, aby wytrenować model (Należy zdefiniować tę funkcję).
Wizualizacja Strat Treningowych:

Rysuje wykres strat treningowych w przebiegu epok, aby monitorować postęp uczenia się modelu.
Wykrywanie Anomalii w Danych Testowych:

Iteruje przez wszystkie pliki testowe w data/test/.
Dla każdego pliku testowego:
Wczytuje i przetwarza dane.
Wykrywa anomalie za pomocą wytrenowanego modelu.
Wizualizuje wyniki.
Wyświetla szczegóły dotyczące wykrytych regionów anomalii.
Zapisanie Modelu:

Zapisuje wytrenowany model do pliku BOMBACLAT_model.pth dla przyszłego użytku bez konieczności ponownego trenowania.
Uwaga:

Niezdefiniowana Funkcja train_model:
Skrypt odnosi się do funkcji train_model, która nie została zdefiniowana w dostarczonym kodzie. Upewnij się, że zaimplementujesz tę funkcję, aby obsłużyć pętlę treningową, w tym przejścia w przód, obliczanie strat, propagację wsteczną i kroki optymalizatora.
Przykładowa Implementacja train_model:

```python
def train_model(model, data_loader, criterion, optimizer, epochs=50):
    model.train()
    training_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(data_loader)
        training_losses.append(avg_loss)
        print(f'Epoka [{epoch+1}/{epochs}], Strata: {avg_loss:.6f}')
    return training_losses
Upewnij się, że ta funkcja jest zawarta w Twoim skrypcie neuron_eeg.py.
```
Wyniki
Po uruchomieniu skryptu otrzymasz:

Wykres Straty Treningowej:

Ilustruje, jak strata modelu zmniejsza się w trakcie epok treningowych, co wskazuje na postęp uczenia się modelu.
Wykresy Wykrywania Anomalii:

Sygnał EEG vs. Sygnał Odtworzony: Pokazuje oryginalny sygnał EEG w kolorze niebieskim i odtworzony sygnał przez model w kolorze zielonym. Odchylenia wskazują potencjalne anomalie.
Błąd Rekonstrukcji: Podkreśla obszary, gdzie błąd rekonstrukcji przekracza próg, oznaczając anomalie.
Regiony Anomalii:

Skrypt wyświetla indeksy początkowe i końcowe każdego wykrytego regionu anomalii oraz jego długość.
Zapisany Model:

Wytrenowany model jest zapisany jako BOMBACLAT_model.pth, umożliwiając jego ponowne użycie bez konieczności ponownego trenowania.
Uwagi
Jakość Danych: Upewnij się, że Twoje dane EEG są czyste i dobrze sformatowane. Brakujące lub uszkodzone dane mogą negatywnie wpływać na wydajność modelu.

Dostosowanie Progu: Parametr threshold_percentile w funkcji detect_anomalies określa czułość wykrywania anomalii. Dostosuj go w zależności od specyficznych wymagań.

Rozmiar Okna: Parametr window_size wpływa na to, jak anomalie są grupowane. Większe okno łączy anomalie w szerszych zakresach czasowych.

Dostosowanie Modelu: Możesz modyfikować architekturę BOMBACLAT (np. liczbę warstw, liczbę neuronów), aby lepiej dopasować model do swoich danych.

Rozważania Wydajnościowe: Trenowanie modeli głębokiego uczenia może być zasobożerne. Upewnij się, że Twój system ma odpowiednie zasoby obliczeniowe.

Dalsze Ulepszenia:

Implementacja bardziej zaawansowanych technik wstępnego przetwarzania danych.
Eksploracja różnych architektur modeli lub hiperparametrów.
Dodanie zbiorów walidacyjnych do monitorowania przeuczenia.
