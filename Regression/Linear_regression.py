#%%
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read dataset into a DataFrame
df = pd.read_csv("C:\Personal\Masters\Masters_work\Study\Y1_S1\HIT140\Assessment_2\HIT_140_Assessment_2_200925\HIT_140_assessment_2\Assignment_3\Linear_regression_datasets\Spring.csv")
# df = pd.read_csv("C:\Personal\Masters\Masters_work\Study\Y1_S1\HIT140\Assessment_2\HIT_140_Assessment_2_200925\HIT_140_assessment_2\Assignment_3\Linear_regression_datasets\Winter.csv")
#%%
"""
BUILD AND EVALUATE A LINEAR REGRESSION MODEL
"""

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1].values
y = df.iloc[:,1].values

# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)
# Plotting the data and the linear regression line
plt.figure(figsize=(10, 6))

# Scatter plot of actual data points
plt.scatter(X_train, y_train, color='blue', label='Training Data', marker='o')
plt.scatter(X_test, y_test, color='green', label='Test Data', marker='x')

# Plotting the regression line
# Create a range of values for x to plot the regression line
x_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_range = model.predict(x_range)
plt.plot(x_range, y_range, color='red', label='Regression Line')

# Adding labels and title
plt.title('Linear Regression Model')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.grid()

# Show the plot
plt.show()

# Optional: Print performance metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
y_max = y_test.max()
y_min = y_test.min()
rmse_norm = rmse / (y_max - y_min)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)



# %%
