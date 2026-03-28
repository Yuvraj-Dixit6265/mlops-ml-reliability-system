import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

#split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

 #save model

joblib.dump(model, "model/model.joblib")


print("Model trained and saved successfully!")

# Save training data for drift comparison
X_train.to_csv("data/train_data.csv", index=False)