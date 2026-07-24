import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score


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


#Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y #so training and testing keeps same amount of fraud
)


#Scaling the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train) #only learning scaling from training data
X_test = scaler.transform(X_test)


print("Training samples: ",X_train.shape)
print("Testing samples: ",X_test.shape)


# Train model
model = LogisticRegression()

model.fit(X_train, y_train) #training the model

# Predictions
y_pred = model.predict(X_test)


#AUPRC
y_prob = model.predict_proba(X_test)[:, 1]
auprc = average_precision_score(y_test, y_prob)
print(f"AUPRC: {auprc}")


# Evaluation - Display results
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

print(f"Confusion Matrix: {confusion_matrix(y_test, y_pred)}")

print(f"Classification Report: {classification_report(y_test, y_pred)}")

