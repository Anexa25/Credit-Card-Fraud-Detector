import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score

from sklearn.ensemble import RandomForestClassifier


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


# Train model - Using logistic regression model

def train_logistic_regression(X_train, X_test, y_train, y_test):
    
    model = LogisticRegression(max_iter=1000)
    
    print("Training Logistic Regression")

    model.fit(X_train, y_train) #training the model

    # Predictions
    y_pred = model.predict(X_test)


    #AUPRC
    y_prob = model.predict_proba(X_test)[:, 1]
    auprc = average_precision_score(y_test, y_prob)


    # Evaluation - Display results
    print("Logistic Regression Results")
    print("-"*30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    print(f"Confusion Matrix: {confusion_matrix(y_test, y_pred)}")

    print(f"Classification Report: {classification_report(y_test, y_pred)}")
    
    print(f"AUPRC: {auprc}")
 
    return model





# Upgraded model - Random Forest Classifier

def train_random_forest(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(
        n_estimators=100, #build 100 decision trees
        random_state=42, #makes results reproducible
        class_weight="balanced", #gives more importance to rare fraud cases during training
        n_jobs=-1
    )


    print("Training Random Forest Model")

    rf_model.fit(X_train,y_train)

    # Predictions
    rf_predictions = rf_model.predict(X_test)
    rf_probabilities = rf_model.predict_proba(X_test)[:,1]


    # Evaluation:
    rf_auprc = average_precision_score(y_test, rf_probabilities)

    print("Random Forest Results")
    print("-"*25)
    print(f"Accuracy: {accuracy_score(y_test, rf_predictions)}")

    print(f"Confusion Matrix: {confusion_matrix(y_test, rf_predictions)}")

    print(f"Classification Report: {classification_report(y_test, rf_predictions)}")

    print(f"AUPRC: {rf_auprc}")
    
    return rf_model
    
    

# Function calls
logistic_model = train_logistic_regression(X_train, X_test, y_train, y_test)

rf_model = train_random_forest(X_train, X_test, y_train, y_test)