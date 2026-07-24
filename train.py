import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("program started")

target_csv = "data/creditcard.csv"
df = pd.read_csv(target_csv)

#Exploring the dataset
###################################
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
####################################


# Seperating features and targets 
X = df.drop("Class", axis=1) # Contains transaction details
y = df["Class"] # Contains the answer for model to learn