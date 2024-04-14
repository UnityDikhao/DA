import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

heights = np.random.normal(170, 10, 10)
weights = np.random.normal(70, 5, 10)
df = pd.DataFrame({'Height': heights, 'Weight': weights})
X=df[['Height']]
y=df['Weight']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
print("X_train:\n",X_train)
print("y_train:\n",y_train)
print("X_test:\n",X_test)
print("y_test:\n",y_test)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Coefficients:",model.coef_)
print("Intercept:",model.intercept_)
