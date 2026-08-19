import pandas as pd

df= pd.read_parquet("1.parquet")
df.to_csv("1.csv", index=False)
