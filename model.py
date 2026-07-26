import joblib
import pandas as pd

# Loading the trained model and trainer
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Successfully loaded model and scaler")


def predict_transaction(transaction):
    
    #Scale transaction
    scaled_transaction = scaler.transform(transaction)
    
    #Make prediction
    prediction = model.predict(scaled_transaction)
    
    #get fraud probability
    probability = model.predict_proba(scaled_transaction)[:,1]
    
    return prediction[0], probability[0]
    
    
    
# # Testing the model

# df = pd.read_csv("data/creditcard.csv")

# #sample = df.drop("Class", axis=1).iloc[[0]]
# fraud_samples = df[df["Class"]==1].drop("Class", axis=1) #isolating a fraud transaction

# correct = 0

# for i in range(len(fraud_samples)):
#     sample = fraud_samples.iloc[[i]]
    
#     result, probability = predict_transaction(sample)
    
#     if result == 1:
#         correct +=1

# # print(f"Prediction: {result}")
# # print(f"Fraud probability: {probability}")
# # print("-"*15)

# print(f"Fraud detected: {correct}")
# print(f"Total fraud cases: {len(fraud_samples)}")
# print(f"detection rate: {correct/len(fraud_samples)}")

#test

if __name__ == "__main__":

    import pandas as pd

    df = pd.read_csv("data/creditcard.csv")

    sample = df.drop("Class", axis=1).iloc[[0]]

    prediction, probability = predict_transaction(sample)

    print("Prediction:", prediction)
    print("Probability:", probability)