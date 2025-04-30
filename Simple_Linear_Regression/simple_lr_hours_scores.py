#Simple Linear Regression Project
#This project uses the hours and scores dataset to predict student scores based on hours studied.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Create a simple dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Scores': [35, 45, 55, 60, 70, 80, 85, 90]
}

df = pd.DataFrame(data)

print(df)

x=df[['Hours']] # Feature matrix (independent variable)
y=df['Scores'] # Target vector (dependent variable) 

# Split the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
print(len(x_train),len(x_test))
print(x_train.shape)
print(y_train.shape)

#Intialize and train the model
model=LinearRegression()
model.fit(x_train,y_train)

# Predict using the test set
y_pred=model.predict(x_test)

# Output model parameters
print(f"Intercept (c): {model.intercept_:.3f}")
print(f"Slope (m): {model.coef_[0]:.3f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test,y_pred):.3f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test,y_pred):.3f}")
print(f"R-squared: {r2_score(y_test,y_pred):.3f}")


plt.figure(figsize=(10, 6))
plt.scatter(x_test,y_test,color='blue',label='Actual')
plt.plot(x_test,y_pred,color='red',label='Prediction Line')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Linear Regression: Hours vs Scores')
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show()


hours = [[9.25]]  # Important: 2D array (same format as X)
predicted_score = model.predict(hours)
print(f"Predicted score: {min(predicted_score[0], 100):.2f}")