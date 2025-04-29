#Simple Linear Regression Project
#This project uses the California housing dataset to predict house values based on median income.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

#1. Load the dataset
housing=fetch_california_housing()

#2.Select one feature: Median Income (column 0)
x=housing.data[:,[0]]
y=housing.target

#3. Split the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#4. Initialize and train the model
model=LinearRegression()
model.fit(x_train,y_train)

#5.Predict using the test set
y_pred=model.predict(x_test)
#6. Output model parameters
print(f"Intercept (c): {model.intercept_:.3f}")
print(f"Slope (m): {model.coef_[0]:.3f}")

# 7. Evaluate model performance
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")

# 8. Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, y_pred, color='red', label='Prediction Line')
plt.xlabel("Median Income (in $10,000s)")
plt.ylabel("House Value (in $100,000s)")
plt.title("Linear Regression: Income vs House Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
