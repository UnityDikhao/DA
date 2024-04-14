import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('nursery.csv')

X = data.drop(columns=['purchases'])
y = data['purchases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting set shape:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Predictions:")
print(predictions[:5])
print("\nActual values:")
print(y_test.head())
