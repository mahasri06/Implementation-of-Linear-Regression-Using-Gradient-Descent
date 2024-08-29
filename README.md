# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1: Start


step 2: Initialize weights (`theta`) to zero.



step 3: Add Bias: Insert a column of ones for the bias term in the feature matrix.


step 4: For each iteration, compute predictions using the current `theta`.


Step 5: Calculate the error as the difference between predictions and actual target values.


Step 6: Update `theta` using gradient descent to minimize the error.


Step 7: Repeat steps 3 to 5 for a specified number of iterations.


Step 8: Apply the trained model to new, scaled data to make predictions.

step 9: End

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: Mahasri P
RegisterNumber:  212223100029
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    # Add a column of ones to X1 to account for the bias term
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1, 1)

    for _ in range(num_iters):
        # Calculate predictions
        predictions = X.dot(theta).reshape(-1, 1)
        # Calculate error
        error = predictions - y.reshape(-1, 1)
        # Update theta using gradient descent
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(error)

    return theta

# Load and prepare data
data = pd.read_csv("C:/Users/admin/Downloads/50_Startups.csv")
X = data.iloc[:, :-2].values.astype(float)
y = data.iloc[:, -1].values.reshape(-1, 1)

# Standardize features and target variable
scaler = StandardScaler()
X1_Scaled = scaler.fit_transform(X)
y_Scaled = scaler.fit_transform(y)

# Learn model parameters
theta = linear_regression(X1_Scaled, y_Scaled)

# Predict for new data
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(1, -1)
new_data_Scaled = scaler.transform(new_data)
prediction = np.dot(np.append(1, new_data_Scaled), theta)
pre = scaler.inverse_transform(prediction.reshape(-1, 1))

print(f"Predicted value: {pre}")


```

## Output:
![Screenshot 2024-08-29 092930](https://github.com/user-attachments/assets/011cac43-beb0-4903-b213-b8d5e15069f6)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
