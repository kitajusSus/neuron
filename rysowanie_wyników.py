
# rysowanie_wyników.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_waveform_with_classifications(data, event_labels, group_size):
    """
    Rysowanie wykresów z klasyfikacjami zdarzeń
    """
    n_channels = data.shape[1]
    n_rows = (n_channels + 2) // 3  # 3 plots per row
    n_cols = min(3, n_channels)
    
    # Create figure with subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle("EEG Waveforms with Event Classifications", fontsize=16)
    
    # Flatten axs array for easier iteration
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]
    
    # Create color palette for events
    unique_labels = np.unique(event_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Plot each channel
    for channel in range(n_channels):
        ax = axs[channel]
        
        # Create DataFrame for seaborn
        df = pd.DataFrame({
            'Sample': range(len(data)),
            'Amplitude': data[:, channel]
        })
        
        # Plot the channel data
        sns.lineplot(data=df, x='Sample', y='Amplitude', ax=ax, alpha=0.7)
        
        # Add colored backgrounds for classifications
        ymin, ymax = ax.get_ylim()
        for i, label in enumerate(event_labels):
            start = i * group_size
            end = (i + 1) * group_size
            if end <= len(data):
                ax.axvspan(start, end, alpha=0.2, color=label_color_map[label],
                          label=f'Event {label}' if channel == 0 else "")
        
        ax.set_title(f'Channel {channel}')
        ax.grid(True)
        if channel == 0:  # Add legend only to first plot
            handles, labels = ax.get_legend_handles_labels()
            # Remove duplicate labels
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove any unused subplots
    for idx in range(n_channels, len(axs)):
        fig.delaxes(axs[idx])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
    plt.show()

def plot_reconstruction(df, reconstructed, loss, dilated_anomalies, labeled_regions, num_regions, threshold):
    """
    Rysowanie wyników rekonstrukcji i wykrywania anomalii
    """
    # Oryginalne dane i odtworzone dane
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
