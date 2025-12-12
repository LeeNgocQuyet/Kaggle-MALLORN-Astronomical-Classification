import pandas as pd
df = pd.read_csv("data/processed/train_final.csv")
print(train_df.head())
print(train_df.describe())
print(train_df.isna().sum())