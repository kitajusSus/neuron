# rysowanie_wyników.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
