import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

data =pd.read_csv("fish_data.csv")

df = pd.DataFrame(data)

x=df[['Length1','Length2','Length3','Height','Width']]
y = df['Weight']

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size = 0.3,random_state=42)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = metrics.mean_squared_error(y_test,y_pred)
smse = np.sqrt(mse)

print("model:\n")
print("mse = ",mse)
print("smse = ",smse)
print("Coefficients:",model.coef_)
print("Intercept:",model.intercept_)