#Q1 a-c

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load the dataset from CSV
df = pd.read_csv('mlr_dataset.csv')

# a) Identify the independent and dependent features
independent_features = df[['AT', 'V', 'AP', 'RH']].values
dependent_feature = df['PE'].values

# Plot each independent feature with the dependent feature
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].scatter(independent_features[:, 0], dependent_feature)
axs[0, 0].set_xlabel('Air Temperature (AT)')
axs[0, 0].set_ylabel('Net Hourly Electrical Energy Output (PE)')

axs[0, 1].scatter(independent_features[:, 1], dependent_feature)
axs[0, 1].set_xlabel('Exhaust Vacuum (V)')
axs[0, 1].set_ylabel('Net Hourly Electrical Energy Output (PE)')

axs[1, 0].scatter(independent_features[:, 2], dependent_feature)
axs[1, 0].set_xlabel('Ambient Pressure (AP)')
axs[1, 0].set_ylabel('Net Hourly Electrical Energy Output (PE)')

axs[1, 1].scatter(independent_features[:, 3], dependent_feature)
axs[1, 1].set_xlabel('Relative Humidity (RH)')
axs[1, 1].set_ylabel('Net Hourly Electrical Energy Output (PE)')

plt.tight_layout()
plt.show()


# b) Perform Multiple Regression
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    independent_features, dependent_feature, test_size=0.2, random_state=42
)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the values of the error for the first five data points
predictions = model.predict(X_test)
errors = y_test - predictions
print("Errors for the first five data points:", errors[:5])

#OUTPUT ANSWER:
#Errors for the first five data points: [-0.41020791 -2.42212215  6.51556    -4.36954566  2.17167134]

# c) Create a plot of observed data against predicted data
plt.scatter(y_test, predictions)
plt.xlabel('Observed Data')
plt.ylabel('Predicted Data')
plt.title('Observed vs Predicted Data')
plt.show()

# d) Perform Multiple Regression with selected features
selected_features = df[['AT', 'V', 'PE']].values

# Fit the model
model_selected = LinearRegression()
model_selected.fit(selected_features[:, :-1], selected_features[:, -1])

# Create a 3D plot with the predicted data hyperplane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of data points
ax.scatter(selected_features[:, 0], selected_features[:, 1], selected_features[:, 2], c='b', marker='o', label='Observed Data')

# Create a meshgrid for the hyperplane
meshgrid_x, meshgrid_y = np.meshgrid(selected_features[:, 0], selected_features[:, 1])
predicted_data_hyperplane = model_selected.predict(np.c_[meshgrid_x.ravel(), meshgrid_y.ravel()]).reshape(meshgrid_x.shape)

# Plot the hyperplane
ax.plot_surface(meshgrid_x, meshgrid_y, predicted_data_hyperplane, alpha=0.5, color='r')

ax.set_xlabel('Air Temperature (AT)')
ax.set_ylabel('Exhaust Vacuum (V)')
ax.set_zlabel('Net Hourly Electrical Energy Output (PE)')

plt.title('3D Plot with Predicted Data Hyperplane')
plt.show()

##############################################################################
#Q5

import cvxpy as cp
import numpy as np

# Define the variables
x = cp.Variable(2)

# Define the objective function
Q = np.array([[6, 2], [2, 8]])
c = np.array([-2, -3])
objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)

# Define the inequality constraints
constraints = [3*x[0] + 2*x[1] <= 6, x[0] + x[1] <= 2, x[0] >= 0, x[1] >= 0]

# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Print the results
print("Optimal value:", problem.value)
print("Optimal solution x:", x.value)

#ANS:
#Optimal value: -0.7045454545454546
#Optimal solution x: [0.22727273 0.31818182]
