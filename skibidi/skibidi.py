import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfiguracja Eksperymentu i Symulacji Danych ---
SAMPLING_RATE = 1000  # Częstotliwość próbkowania w Hz
SEGMENT_DURATION_SECONDS = 5  # Czas trwania jednego segmentu (neutralny, emocja) w sekundach
SAMPLES_PER_SEGMENT = SAMPLING_RATE * SEGMENT_DURATION_SECONDS

NUM_MUSCLES = 3
MUSCLE_NAMES = [f'Mięsień {chr(65+i)}' for i in range(NUM_MUSCLES)] # Mięsień A, Mięsień B, Mięsień C

EMOTIONS = ['Radość', 'Smutek', 'Złość'] # Emocje do analizy, bez Neutralnego
EMOTION_TRIALS_PER_EMOTION = 4 # Liczba powtórzeń dla każdej emocji

# --- Funkcja Symulacji Danych EMG ---
def simulate_emg_data():
    """
    Symuluje dane EMG dla stanu neutralnego i różnych emocji.
    Generuje dane z różnym "szumem" (aktywnością) dla poszczególnych faz.
    """
    all_data = []
    all_labels = []

    # 1. Dane dla stanu Neutralnego (bazowego)
    # Niska aktywność
    neutral_data = np.random.normal(loc=0, scale=0.05, size=(SAMPLES_PER_SEGMENT, NUM_MUSCLES))
    all_data.append(neutral_data)
    all_labels.extend(['Neutral_Baseline'] * SAMPLES_PER_SEGMENT)

    # 2. Dane dla poszczególnych emocji (z 4 powtórzeniami każda)
    # Symuluje, że niektóre mięśnie są bardziej aktywne dla danych emocji
    emotion_scales = {
        'Radość':   {'Mięsień A': 0.2, 'Mięsień B': 0.1, 'Mięsień C': 0.07},
        'Smutek':   {'Mięsień A': 0.08, 'Mięsień B': 0.25, 'Mięsień C': 0.1},
        'Złość':    {'Mięsień A': 0.15, 'Mięsień B': 0.1, 'Mięsień C': 0.3}
    }

    for emotion in EMOTIONS:
        for i in range(EMOTION_TRIALS_PER_EMOTION):
            trial_data = np.zeros((SAMPLES_PER_SEGMENT, NUM_MUSCLES))
            for j, muscle in enumerate(MUSCLE_NAMES):
                # Generujemy dane dla mięśnia z odpowiednią "aktywnością"
                scale = emotion_scales[emotion][muscle] + np.random.uniform(-0.02, 0.02) # Dodajemy trochę losowości
                trial_data[:, j] = np.random.normal(loc=0, scale=scale, size=SAMPLES_PER_SEGMENT)

                # Dodajemy symulowane "skurcze" - krótkie, większe amplitudy
                if np.random.rand() > 0.3: # Szansa na skurcz
                    start_idx = np.random.randint(0, SAMPLES_PER_SEGMENT - SAMPLING_RATE // 2)
                    end_idx = start_idx + SAMPLING_RATE // 2 # 0.5 sekundy skurczu
                    burst_amplitude = np.random.uniform(0.5, 1.5) * scale * 5
                    t = np.linspace(0, 0.5, SAMPLING_RATE // 2)
                    burst_signal = burst_amplitude * np.sin(2 * np.pi * 10 * t) * np.exp(-10 * t) # Fala zanikająca
                    trial_data[start_idx:end_idx, j] += burst_signal[:len(trial_data[start_idx:end_idx, j])]

            all_data.append(trial_data)
            all_labels.extend([f'{emotion}_{i+1}'] * SAMPLES_PER_SEGMENT)

    # Tworzenie DataFrame
    flat_data = np.vstack(all_data)
    df = pd.DataFrame(flat_data, columns=MUSCLE_NAMES)
    df['timestamp'] = np.arange(len(df)) / SAMPLING_RATE
    df['label'] = all_labels

    return df

# --- Funkcja Obliczania Wskaźnika Aktywności ---
def calculate_activity_indicator(data_segment, indicator_type='rms'):
    """
    Oblicza wybrany wskaźnik aktywności dla segmentu danych.
    :param data_segment: Fragment danych EMG dla jednego mięśnia.
    :param indicator_type: 'rms' (domyślny), 'variance', 'peak_to_peak'.
    :return: Wartość wskaźnika.
    """
    if indicator_type == 'rms':
        return np.sqrt(np.mean(data_segment**2))
    elif indicator_type == 'variance':
        return np.var(data_segment)
    elif indicator_type == 'peak_to_peak':
        return np.max(data_segment) - np.min(data_segment)
    else:
        raise ValueError("Nieobsługiwany typ wskaźnika. Wybierz 'rms', 'variance' lub 'peak_to_peak'.")

# --- Główna Funkcja Przetwarzająca Dane ---
def process_emg_emotion_data(df_emg, indicator_type='rms'):
    """
    Przetwarza dane EMG, oblicza aktywność mięśni względem stanu neutralnego
    i uśrednia wyniki dla każdej emocji.
    :param df_emg: DataFrame z danymi EMG i kolumną 'label'.
    :param indicator_type: Typ wskaźnika aktywności do użycia ('rms' lub 'variance').
    :return: DataFrame z uśrednionymi wynikami aktywności.
    """
    # 1. Wyodrębnienie danych neutralnych (bazowych)
    neutral_data_df = df_emg[df_emg['label'] == 'Neutral_Baseline']
    if neutral_data_df.empty:
        raise ValueError("Brak segmentu 'Neutral_Baseline' w danych.")

    # Obliczanie wskaźników dla stanu neutralnego dla każdego mięśnia
    neutral_indicators = {}
    for muscle in MUSCLE_NAMES:
        neutral_indicators[muscle] = calculate_activity_indicator(neutral_data_df[muscle].values, indicator_type)
        if neutral_indicators[muscle] == 0:
            # Dodaj małą wartość, aby uniknąć dzielenia przez zero, jeśli RMS jest idealnie 0 (np. dla symulowanych danych)
            neutral_indicators[muscle] = 1e-9

    # 2. Obliczanie aktywności dla każdej emocji i każdego powtórzenia
    raw_emotion_scores = {muscle: {emotion: [] for emotion in EMOTIONS} for muscle in MUSCLE_NAMES}

    for label in df_emg['label'].unique():
        if label.startswith(tuple(EMOTIONS)): # Jeśli label odpowiada jednej z emocji
            emotion_name = label.split('_')[0]
            
            segment_data_df = df_emg[df_emg['label'] == label]

            for muscle in MUSCLE_NAMES:
                current_indicator = calculate_activity_indicator(segment_data_df[muscle].values, indicator_type)
                
                # Obliczanie procentowego wzrostu względem neutralnego
                relative_score = ((current_indicator - neutral_indicators[muscle]) / neutral_indicators[muscle]) * 100
                raw_emotion_scores[muscle][emotion_name].append(relative_score)

    # 3. Uśrednianie wyników dla każdego mięśnia i emocji
    averaged_scores = []
    for muscle in MUSCLE_NAMES:
        for emotion in EMOTIONS:
            avg_score = np.mean(raw_emotion_scores[muscle][emotion])
            averaged_scores.append({
                'Mięsień': muscle,
                'Emocja': emotion,
                'Aktywność (%)': avg_score
            })
    
    results_df = pd.DataFrame(averaged_scores)
    return results_df

# --- Funkcja Wizualizacji Wyników ---
def plot_results(results_df, indicator_type='rms'):
    """
    Wizualizuje uśrednione wyniki aktywności mięśni dla każdej emocji.
    """
    plt.figure(figsize=(12, 7))
    sns.barplot(data=results_df, x='Emocja', y='Aktywność (%)', hue='Mięsień', palette='viridis')
    plt.title(f'Uśredniona Aktywność Mięśni dla Emocji (Wskaźnik: {indicator_type.upper()} - % Wzrostu vs Neutralny)', fontsize=16)
    plt.xlabel('Emocja', fontsize=12)
    plt.ylabel(f'Procentowy Wzrost Aktywności ({indicator_type.upper()}) vs Neutralny', fontsize=12)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Linia bazowa dla 0% wzrostu
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Mięsień', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --- Główna Sekwencja Wykonania ---
if __name__ == "__main__":
    print("--- Symulacja i Analiza Aktywności Mięśni w Reakcji na Emocje ---")

    # 1. Symulacja danych EMG (zamiast wczytywania z CSV)
    print("Generowanie symulowanych danych EMG...")
    emg_df = simulate_emg_data()
    print(f"Wygenerowano dane o rozmiarze: {emg_df.shape}")
    print("Pierwsze 5 wierszy danych:")
    print(emg_df.head())
    print("\nUnikalne etykiety segmentów (przykładowe):")
    print(emg_df['label'].unique())
    print("-" * 50)

    # 2. Przetwarzanie danych i obliczanie aktywności
    print("Przetwarzanie danych i obliczanie aktywności mięśni...")
    # Możesz zmienić 'rms' na 'variance' lub 'peak_to_peak'
    analysis_indicator = 'rms' 
    results_df = process_emg_emotion_data(emg_df, indicator_type=analysis_indicator)
    
    print("\nUśrednione wyniki aktywności mięśni (procentowy wzrost vs neutralny):")
    print(results_df)
    print("-" * 50)

    # 3. Wizualizacja wyników
    print("\nWizualizacja wyników...")
    plot_results(results_df, indicator_type=analysis_indicator)
    print("-" * 50)

    # 4. Interpretacja Wyników (tekstowa)
    print("\n--- Interpretacja Wyników ---")
    print("Na podstawie symulowanych danych i obliczonego procentowego wzrostu aktywności (RMS) względem stanu neutralnego,")
    print("możemy zaobserwować zaangażowanie poszczególnych mięśni w wyrażanie zdefiniowanych emocji:")
    
    for emotion in EMOTIONS:
        print(f"\nEmocja: {emotion}")
        emotion_results = results_df[results_df['Emocja'] == emotion].sort_values(by='Aktywność (%)', ascending=False)
        
        # Określanie mięśnia najbardziej zaangażowanego
        most_engaged_muscle = emotion_results.iloc[0]['Mięsień']
        most_engaged_activity = emotion_results.iloc[0]['Aktywność (%)']

        print(f"  - Mięsień najbardziej zaangażowany: {most_engaged_muscle} (wzrost o {most_engaged_activity:.2f}%)")
        print("  - Aktywność dla wszystkich mięśni w tej emocji:")
        for index, row in emotion_results.iterrows():
            print(f"    - {row['Mięsień']}: {row['Aktywność (%)']:.2f}%")
        
        # Ogólna ocena zaangażowania
        if most_engaged_activity > 50: # Próg dla "dużego" zaangażowania
            print(f"  -> Ogólnie dla emocji '{emotion}' obserwujemy znaczące zaangażowanie mięśnia {most_engaged_muscle}.")
        elif most_engaged_activity > 10:
            print(f"  -> Aktywność dla emocji '{emotion}' wskazuje na pewne zaangażowanie mięśni, szczególnie {most_engaged_muscle}.")
        else:
            print(f"  -> Aktywność dla emocji '{emotion}' jest niska lub zbliżona do stanu neutralnego dla większości mięśni.")
            
    print("\n DANE SYMULOWANE LOSOWANE Z PARAMETRAMI 'TENDENCJI ' DLA ROZNYCH EMOCJI DLA MIESNI")
    print("SKIBIDI DSKIDBII SKIBIDI , aby wykazać różnice.")
