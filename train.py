import pandas as pd

print("program started")

target_csv = "data/creditcard.csv"
df = pd.read_csv(target_csv)

# First 5 rows
print(df.head())

# Size of dataset 
print(len(df))

# No. of rows and columns
print(df.shape)

# Dataset columns
print(df.columns)

#Count of each Class value
print(df["Class"].value_counts())
