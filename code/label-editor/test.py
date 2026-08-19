import pandas as pd
df = pd.read_parquet("/home/billa-dakait/Desktop/test/Features/slr/specific/bad/18.parquet")
print(df["pelvis_center_y"].describe())
print(df.loc[df["pelvis_center_y"].idxmin(), ["frame","pelvis_center_y"]])