import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#CreatethePosition_Salariesdataset
data={'YearsExperience':[1,2,3,4,5,6,7,8,9,10],
 'Salary':[50000,60000,70000,80000,90000,100000,110000,120000,130000,140000]}
df=pd.DataFrame(data)

#Identify the independent and target variables
X=df[['YearsExperience']]
y=df['Salary']

#Splitthevariablesintotrainingandtestingsetswitha7:3ratio
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#Print the training and testing sets
print("X_train:\n",X_train)
print("y_train:\n",y_train)
print("X_test:\n",X_test)
print("y_test:\n",y_test)

#implemenation of linear regression model
model=LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

#Print the coefficients and intercept
print("Coefficients:",model.coef_)
print("Intercept:",model.intercept_)