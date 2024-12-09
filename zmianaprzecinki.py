import pandas as pd


plik = 'marekmkole1'
# Wczytaj dane z tabulatorami
df = pd.read_csv(plik+'.csv', sep='\t')

# Zapisz dane ze średnikami
df.to_csv( plik+'nowy.csv', sep=';', index=False, header=0)
