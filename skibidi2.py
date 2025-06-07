import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert
import xml.etree.ElementTree as ET
import re

# --- Konfiguracja Plików i Parametry ---
CSV_FILE = '/content/kaimordai.csv'
TAG_FILE = '/content/kai-tw.obci.tag'
FS = 1000  # Częstotliwość próbkowania EMG w Hz (częsta dla EMG)

# --- Funkcja do Wczytywania i Parsowania Tagów ---
def load_and_parse_tags(tag_filepath):
    """
    Wczytuje i parsuje plik tagów .obci.tag, zwracając DataFrame.
    """
    with open(tag_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        tag_lines = [line.strip() for line in f if "<tag " in line]

    tag_data = []
    for line in tag_lines:
        try:
            tag = ET.fromstring(line)
            name = tag.attrib['name']
            position = float(tag.attrib['position'])
            duration = float(tag.attrib['length'])
            
            # wypierdol numerki i ".bmp" z końca (np. "zlosc1.bmp" -> "zlosc")
            emotion = re.sub(r'\d+\.bmp', '', name).strip().lower()
            # Dodatkowe czyszczenie, np. "neutralna_baseline" -> "neutralna"
            if "baseline" in emotion:
                emotion = "neutralna"
            
            tag_data.append({'emotion': emotion, 'start_sec': position, 'duration_sec': duration})
        except ET.ParseError:
            print(f"Ostrzeżenie: Nie można sparsować linii: {line}")
            continue
    return pd.DataFrame(tag_data)

# --- Funkcja do Obliczania Wskaźników Aktywności dla Segmentu ---
def calculate_emg_features(signal_segment):
    """
    Oblicza wariancję i średnią amplitudę obwiedni sygnału EMG dla danego segmentu.
    """
    if len(signal_segment) < 10:  # Minimalna długość segmentu, aby uniknąć błędów
        return {'variance': np.nan, 'amplitude': np.nan}

    # Wariancja sygnału
    variance = np.var(signal_segment)

    # Obwiednia sygnału (z użyciem transformaty Hilberta) i średnia amplituda
    envelope = np.abs(hilbert(signal_segment))
    amplitude = np.mean(envelope)

    return {'variance': variance, 'amplitude': amplitude}

# --- Główna Funkcja Analizy Danych EMG dla Emocji ---
def analyze_emg_for_emotions(csv_file, tag_file, fs):
    """
    Przeprowadza pełną analizę danych EMG pod kątem emocji.
    Wczytuje dane, parsuje tagi, oblicza wariancję i amplitudę,
    a następnie procentowy wzrost względem stanu neutralnego.
    """
    # 1. Wczytanie danych EMG
    df_emg = pd.read_csv(csv_file, header=None) # Zakładam brak nagłówka
    # Automatyczne wykrycie kanałów EMG (np. kolumny 0, 1, 2)
    # Możemy nadać im bardziej opisowe nazwy, jeśli wiemy, co reprezentują
    channels = [f'Mięsień {i+1}' for i in range(df_emg.shape[1])]
    df_emg.columns = channels
    print(f"Wczytano dane EMG z {df_emg.shape[1]} kanałów: {channels}")

    # 2. Parsowanie tagów
    tags_df = load_and_parse_tags(tag_file)
    print(f"Wczytano {len(tags_df)} tagów.")
    print("Unikalne emocje w tagach:", tags_df['emotion'].unique())

    # 3. Filtracja poprawnych znaczników
    # Upewniamy się, że tagi mieszczą się w zakresie danych EMG
    max_idx = len(df_emg)
    valid_tags = tags_df[
        (tags_df['start_sec'] * fs < max_idx) &
        ((tags_df['start_sec'] + tags_df['duration_sec']) * fs <= max_idx) &
        (tags_df['duration_sec'] > 0)
    ].copy()
    print(f"Pozostało {len(valid_tags)} poprawnych tagów po filtracji.")

    # 4. Obliczenie miar dla każdego segmentu
    raw_results = {emotion: {ch: [] for ch in channels} for emotion in valid_tags['emotion'].unique()}

    for _, row in valid_tags.iterrows():
        start_sample = int(row['start_sec'] * fs)
        end_sample = int((row['start_sec'] + row['duration_sec']) * fs)
        emotion = row['emotion']

        if start_sample >= end_sample: # Pomiń segmenty o zerowej lub ujemnej długości
            continue

        for ch in channels:
            signal_segment = df_emg[ch].iloc[start_sample:end_sample].values
            features = calculate_emg_features(signal_segment)
            if pd.notna(features['variance']) and pd.notna(features['amplitude']):
                raw_results[emotion][ch].append(features)

    # 5. Uśrednianie wyników dla każdej emocji i mięśnia
    averaged_features = {}
    for emotion, channels_data in raw_results.items():
        averaged_features[emotion] = {}
        for ch, feature_list in channels_data.items():
            if feature_list:
                avg_var = np.mean([f['variance'] for f in feature_list])
                avg_amp = np.mean([f['amplitude'] for f in feature_list])
                averaged_features[emotion][ch] = {'variance': avg_var, 'amplitude': avg_amp}
            else:
                averaged_features[emotion][ch] = {'variance': np.nan, 'amplitude': np.nan}

    # 6. Obliczenie procentowego wzrostu względem emocji 'neutralna'
    results_list = []
    
    # Pobierz wartości bazowe dla "neutralna"
    neutral_data = averaged_features.get('neutralna', {})
    if not neutral_data:
        print("Ostrzeżenie: Brak tagu 'neutralna' lub danych dla niego. Nie można obliczyć procentowego wzrostu.")
        return pd.DataFrame(), pd.DataFrame() # Zwróć puste DataFrame

    for emotion, channels_data in averaged_features.items():
        if emotion == 'neutralna':
            continue # Pomiń samą emocję neutralną w wynikach wzrostu

        for ch, features in channels_data.items():
            # Wartości bazowe dla danego mięśnia z 'neutralna'
            neutral_var_base = neutral_data.get(ch, {}).get('variance', np.nan)
            neutral_amp_base = neutral_data.get(ch, {}).get('amplitude', np.nan)
            
            # Obliczenie procentowego wzrostu
            percent_var_increase = np.nan
            if pd.notna(features['variance']) and pd.notna(neutral_var_base) and neutral_var_base != 0:
                percent_var_increase = ((features['variance'] - neutral_var_base) / neutral_var_base) * 100
            
            percent_amp_increase = np.nan
            if pd.notna(features['amplitude']) and pd.notna(neutral_amp_base) and neutral_amp_base != 0:
                percent_amp_increase = ((features['amplitude'] - neutral_amp_base) / neutral_amp_base) * 100
            
            results_list.append({
                'Emocja': emotion,
                'Mięsień': ch,
                'Procentowy Wzrost Wariancji': percent_var_increase,
                'Procentowy Wzrost Amplitudy': percent_amp_increase
            })
    
    results_df = pd.DataFrame(results_list)
    
    # Dodatkowe DataFrame dla surowych średnich (przydatne do weryfikacji)
    raw_avg_df = pd.DataFrame([
        {'Emocja': em, 'Mięsień': ch, 'Średnia Wariancja': f.get('variance'), 'Średnia Amplituda': f.get('amplitude')}
        for em, ch_data in averaged_features.items()
        for ch, f in ch_data.items()
    ])

    return results_df, raw_avg_df

# --- Funkcja Wizualizacji Wyników ---
def plot_emg_results(results_df, metric_col, title_suffix, ylabel):
    """
    Wizualizuje uśrednione wyniki aktywności mięśni dla każdej emocji.
    """
    if results_df.empty:
        print(f"Brak danych do wykreślenia dla {metric_col}.")
        return

    plt.figure(figsize=(12, 7))
    sns.barplot(data=results_df, x='Emocja', y=metric_col, hue='Mięsień', palette='viridis')
    plt.title(f'Zaangażowanie Mięśni Twarzy – {title_suffix}', fontsize=16)
    plt.xlabel('Emocja', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Linia bazowa dla 0% wzrostu
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Mięsień', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --- Główna Sekwencja Wykonania dla Zadania z Emocjami ---
if __name__ == "__main__":
    print("--- Analiza Aktywności Mięśni Twarzy w Reakcji na Emocje ---")

    # Przeprowadzenie analizy
    emotions_results_df, raw_avg_features_df = analyze_emg_for_emotions(CSV_FILE, TAG_FILE, FS)

    if not emotions_results_df.empty:
        print("\n--- Uśrednione Wyniki (Procentowy Wzrost vs Neutralna) ---")
        print(emotions_results_df)

        print("\n--- Wizualizacja Wyników ---")
        plot_emg_results(emotions_results_df, 'Procentowy Wzrost Wariancji',
                          'Procentowy Wzrost Wariancji vs Neutralna',
                          'Procentowy Wzrost Wariancji (%)')
        plot_emg_results(emotions_results_df, 'Procentowy Wzrost Amplitudy',
                          'Procentowy Wzrost Amplitudy Obwiedni vs Neutralna',
                          'Procentowy Wzrost Amplitudy Obwiedni (%)')
        
        print("\n--- Interpretacja Wyników ---")
        print("Na podstawie analizy sygnału EMG mięśni twarzy, obliczono procentowy wzrost")
        print("wariancji i amplitudy obwiedni sygnału w odniesieniu do stanu neutralnego.")
        print("Pozwala to ocenić zaangażowanie poszczególnych mięśni w wyrażanie emocji.")

        # Przykładowa interpretacja (dostosuj do swoich danych)
        for emotion in emotions_results_df['Emocja'].unique():
            print(f"\nEmocja: {emotion.capitalize()}")
            emotion_data = emotions_results_df[emotions_results_df['Emocja'] == emotion]
            
            # Sortowanie według wzrostu wariancji
            variance_sorted = emotion_data.sort_values(by='Procentowy Wzrost Wariancji', ascending=False)
            print("  Wzrost Wariancji:")
            for _, row in variance_sorted.iterrows():
                print(f"    - {row['Mięsień']}: {row['Procentowy Wzrost Wariancji']:.2f}%")
            
            # Sortowanie według wzrostu amplitudy
            amplitude_sorted = emotion_data.sort_values(by='Procentowy Wzrost Amplitudy', ascending=False)
            print("  Wzrost Amplitudy:")
            for _, row in amplitude_sorted.iterrows():
                print(f"    - {row['Mięsień']}: {row['Procentowy Wzrost Amplitudy']:.2f}%")

            # Ogólna ocena zaangażowania dla danej emocji
            # Tutaj możesz dodać specyficzne wnioski na podstawie tego, który mięsień (kanał)
            # wykazał największy wzrost dla danej emocji.
            # Np. "Jeśli Mięsień 1 jest najbardziej aktywny dla radości..."
            most_active_var = variance_sorted.iloc[0]['Mięsień']
            max_var_increase = variance_sorted.iloc[0]['Procentowy Wzrost Wariancji']
            
            most_active_amp = amplitude_sorted.iloc[0]['Mięsień']
            max_amp_increase = amplitude_sorted.iloc[0]['Procentowy Wzrost Amplitudy']

            print(f"  -> Ogólnie dla emocji '{emotion.capitalize()}' największy wzrost aktywności (wariancji) odnotowano w {most_active_var} ({max_var_increase:.2f}%), a amplitudy w {most_active_amp} ({max_amp_increase:.2f}%).")
            # Możesz dalej rozwinąć tę interpretację na podstawie, które mięśnie powinny być aktywne dla danych emocji
            # (np. Mięśnie podnoszące kąciki ust dla radości, etc. - wymaga wiedzy domenowej).

    else:
        print("\nAnaliza nie powiodła się lub brak wyników do interpretacji.")

    print("\n--- Koniec analizy Zadan:ia z Emocjami ---")
