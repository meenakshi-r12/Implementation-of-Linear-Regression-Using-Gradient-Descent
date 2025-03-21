# Implementation-of-Linear-Regression-Using-Gradient-Descent

# AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

# Program:
/*
Program to implement the linear regression using gradient descent.
Developed by:Meenakshi.R 
RegisterNumber: 212224220062
*/
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1, y, learning_rate=0.01, num_iters=100):
    x = np.c_[np.ones(len(x1)), x1]
    theta = np.zeros(x.shape[1]).reshape(-1,1)
    for i in range(num_iters):
        predictions = (x).dot(theta).reshape(-1,1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(x1)) * x.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv',header=None)
print(data.head())
x = (data.iloc[1:, :-2].values)
print(x)
x1=x.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_scaled = scaler.fit_transform(x)
y1_scaled = scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
theta = linear_regression(x1_scaled, y1_scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```
# Output:

![image](https://github.com/user-attachments/assets/7cecb65b-7c63-4c9e-9fee-08d672eb645d)

# Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
