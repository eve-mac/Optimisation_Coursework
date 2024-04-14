#Q1B

import numpy as np
from scipy.optimize import linprog

# Coefficients of the objective function to be minimised
c = [-5, -3, -4]

# Coefficients of the inequality constraints
A = [[3, 6, 2],
     [6, -7, 1],
     [1, 2, 1]]

# Right-hand side of the inequality constraints
b = [12, 1, 4]

# Bounds for x1, x2, x3
x_bounds = [(0, None), (0, None), (0, None)]

# Solve the linear program
result = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

if result.success:
    print("Optimal point (x1, x2, x3):", result.x)
    print("Optimal value of Z:", -result.fun)
else:
    print("Optimization failed. Check constraints or initial values.")

########
#ANSWER:
#Optimal point (x1, x2, x3): [0.         0.33333333 3.33333333]
#Optimal value of Z: 14.333333333333334

##############################################################################
#Q1D

import numpy as np
from scipy.optimize import linprog

# Coefficients of the objective function to be minimized
c = [-5, -3, -4, 0, 0, 0]  # Include coefficients for slack variables

# Coefficients of the equality constraints (including slack variables)
A_eq = [
    [3, 6, 2, 1, 0, 0],
    [6, -7, 1, 0, 1, 0],
    [1, 2, 1, 0, 0, 1]
]

# Right-hand side of the equality constraints
b_eq = [12, 1, 4]

# Bounds for x1, x2, x3, s1, s2, s3
x_bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]

# Solve the linear program
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')

if result.success:
    optimal_point = result.x[:3]  # Extract x1, x2, x3 from the result
    optimal_value = -result.fun  # Correct the negated value for minimization
    print("Optimal point (x1, x2, x3):", optimal_point)
    print("Optimal value of Z':", optimal_value)
else:
    print("Optimization failed. Check constraints or initial values.")

########
#ANSWER:
#Optimal point (x1, x2, x3): [0.         0.33333333 3.33333333]
#Optimal value of Z': 14.333333333333334

##############################################################################
#Q3E

import numpy as np

# Data points
data_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Calculate sums
n = len(data_points)
sum_x = np.sum(data_points[:, 0])
sum_y = np.sum(data_points[:, 1])
sum_xy = np.sum(data_points[:, 0] * data_points[:, 1])
sum_x_squared = np.sum(data_points[:, 0] ** 2)

# Calculate gradient (m) and y-intercept (c)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
c = (sum_y - m * sum_x) / n

# Print results
print(f"Slope (m): {m}")
print(f"Y-intercept (c): {c}")

# Define the best-fitting line equation
def best_fit_line(x):
    return m * x + c

# Test the equation with chosen point
x_values = [0, 1]
for x in x_values:
    y = best_fit_line(x)
    print(f"For x = {x}, y = {y}")

#ANSWER
#Slope (m): 0.0
#Y-intercept (c): 0.5
