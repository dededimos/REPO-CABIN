import pandas as pd

df = pd.read_csv('cabin_prices_clean.csv')
mask = df.apply(lambda r: r.astype(str).str.contains('90').any(), axis=1)
print(df[mask].head(20).to_string())