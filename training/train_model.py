import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("../data/PS_20174392719_1491204439457_log.csv")

# Reduce dataset size (sampling)
fraud = df[df['isFraud'] == 1]
non_fraud = df[df['isFraud'] == 0].sample(n=200000, random_state=42)

df = pd.concat([fraud, non_fraud])

# Drop unnecessary columns
df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)

# Encode categorical column
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Split features & target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Linear SVM
model = LinearSVC(max_iter=5000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model ONLY (important)
with open("../flask/payments.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")