import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = {'position':['ceo','manager','employee'],
        'level':[1,2,3],
        'salary':[100000,10000,1000]}

df = pd.DataFrame(data)

x = df[['level']]
y = df['salary']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

print("x_train : \n",x_train)
print("x_test : \n",x_test)
print("y_train : \n",y_train)
print("y_test : \n",y_test)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

#Print the coefficients and intercept
print("Coefficients:",model.coef_)
print("Intercept:",model.intercept_)