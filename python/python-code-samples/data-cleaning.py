import pandas as pd

df = pd.read_csv('animals.csv')
print(df.head())
df_clean = df.copy()
df_clean['Animal'] = df_clean['Animal'].str[2:]
df_clean['Body weight (kg)'] = df_clean['Body weight (kg)'].str.replace('!', '.')
df_clean['Brain weight (g)'] = df_clean['Brain weight (g)'].str.replace('!', '.')
df_clean.head()
