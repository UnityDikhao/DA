
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np


np.random.seed(42)

data = {
    'User ID': range(1, 501),
    'Gender': np.random.choice(['Male', 'Female'], size=500),
    'Age': np.random.randint(18, 65, 500),
    'EstimatedSalary': np.random.uniform(30000, 120000, 500),
    'Purchased': np.random.choice([0, 1], size=500)
}

user_df = pd.DataFrame(data)

X = user_df[['Age', 'EstimatedSalary']]
y = user_df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Model Evaluation:")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
