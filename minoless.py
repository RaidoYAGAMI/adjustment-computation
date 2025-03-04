import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
import random
from functions import calculate_angle, decimal_to_dms, dms_to_decimal, calculate_parameters, calculate_side_design_matrix


# Part1: Build True Values and observations

# Define the coordinates of the six points
points = {
    0: (0, 0),
    1: (4, 3),
    2: (0, 4),
    3: (2, 6),
    4: (-2, 8),
    5: (-4, 3)
}

# List of pairs of points to calculate angles
# angle pairs with k,i,j
pairs = [
    (0, 1, 2), (2, 0, 1), (1, 2, 0), (2, 1, 3), (3, 2, 1),
    (1, 3, 2), (2, 3, 4), (4, 2, 3), (3, 4, 2), (5, 2, 4),
    (4, 5, 2), (2, 4, 5), (5, 0, 2), (2, 5, 0), (0, 2, 5)
]

# Calculate all the angles
alpha0 = math.atan2(points[1][1], points[1][0])# Access y as points[1][1] and x as points[1][0]
tanalpha0 = points[1][1] / points[1][0]
# Initialize alpha
alpha = np.zeros(15, dtype=float)  # Ensure floating-point type
# Loop through the pairs to calculate angles
for i, (a, b, c) in enumerate(pairs):
    alpha[i] = calculate_angle(
        points[a][0], points[a][1],  # Point A
        points[b][0], points[b][1],  # Point B
        points[c][0], points[c][1]   # Point C
    )

# convert all the angles into dms format
np.random.seed(666666)
sigma_radians = math.radians(30/3600) #30"
normal_errors = np.random.normal(0, sigma_radians, 15)

alpha_with_errors = alpha + normal_errors

# Calculate all the sides
side_pairs = [
    (0, 1), (1, 3), (3, 4), (4, 5), (5, 0)
]
sides = np.zeros(5, dtype=float)
for i, (a,b) in enumerate(side_pairs):
    sides[i] = math.sqrt((points[b][0]-points[a][0])**2+(points[b][1]-points[a][1])**2)


sigma_sides = 0.01  # 1 cm
side_normal_errors = np.random.normal(0, sigma_sides, 5)
side_with_errors = sides + side_normal_errors

point_errors = np.random.normal(0,0.01,12)
index_to_column = {
    0: (0, 1),  # x0, y0
    1: (2, 3),  # x1, y1
    2: (4, 5),  # x2, y2
    3: (6, 7),  # x3, y3
    4: (8, 9),  # x4, y4
    5: (10, 11)  # x5, y5
}


# Set input points
for point_index, (col_x, col_y) in index_to_column.items():
    # Ensure x is array-like and access its elements
    dx = point_errors[col_x]  # Correction for x
    dy = point_errors[col_y]  # Correction for y
    x_old, y_old = points[point_index]
    points[point_index] = (x_old + dx, y_old + dy)






x_new = np.zeros(12, dtype=float)

# start iteration
sigma0 = 1
# set original variance
variances = [1.5**2, 2**2]
Q_alpha = np.zeros([15,15], dtype=float)
# Create a diagonal matrix with 2's on the diagonal
Q_alpha = np.diag([2] * 15)
Q_alpha[0][3]=Q_alpha[3][0]=-1
Q_alpha[1][12]=Q_alpha[12][1]=-1
Q_alpha[2][14]=Q_alpha[14][2]=-1
Q_alpha[4][7]=Q_alpha[7][4]=-1
Q_alpha[5][6]=Q_alpha[6][5]=-1
Q_alpha[7][9]=Q_alpha[9][7]=-1
Q_alpha[8][11]=Q_alpha[11][8]=-1
Q_alpha[9][14]=Q_alpha[14][9]=-1
Q_alpha[10][13]=Q_alpha[13][10]=-1

Q_sides = np.zeros([5,5], dtype=float)
Q_sides = np.diag([1] * 5)

P_angle = variances[0] * np.linalg.inv(Q_alpha)
P_side = variances[1] * np.linalg.inv(Q_sides)
P = np.block([[P_angle, np.zeros((P_angle.shape[0], P_side.shape[1]))],
              [np.zeros((P_side.shape[0], P_angle.shape[1])), P_side]])

D = sigma0**2 * P
i_num = 0


# MINOLESS doesn't need to iterate
# MINOLESS process
# Build observation
alpha_corrected = np.zeros(15, dtype=float)
# Loop through the pairs to calculate angles
for i, (a, b, c) in enumerate(pairs):
    alpha_corrected[i] = calculate_angle(
        points[a][0], points[a][1],  # Point A
        points[b][0], points[b][1],  # Point B
        points[c][0], points[c][1]  # Point C
    )
# Correct Y and convert it into Radians
Y_alpha = alpha_with_errors - alpha_corrected

sides_corrected = np.zeros(5, dtype=float)
for i, (a, b) in enumerate(side_pairs):
    sides_corrected[i] = math.sqrt((points[b][0] - points[a][0]) ** 2 + (points[b][1] - points[a][1]) ** 2)
Y_side = side_with_errors - sides_corrected

Y = np.concatenate((Y_alpha, Y_side), axis=0)


# set design matrix
B_alpha = np.zeros([15, 12], dtype=float)
# Mapping of point indices to columns in B
# Iterate over pairs and populate B
for m, (k, i, j) in enumerate(pairs):
    # Extract coordinates
    xk, yk = points[k]
    xi, yi = points[i]
    xj, yj = points[j]
    # Calculate parameters
    parameters = calculate_parameters(xk, yk, xi, yi, xj, yj)
    # Map corrections for k, i, and j if they are in index_to_column
    if k in index_to_column:
        col_x, col_y = index_to_column[k]
        B_alpha[m, col_x] = parameters[0]  # dx correction for k
        B_alpha[m, col_y] = parameters[1]  # dy correction for k
    if i in index_to_column:
        col_x, col_y = index_to_column[i]
        B_alpha[m, col_x] = parameters[2]  # dx correction for i
        B_alpha[m, col_y] = parameters[3]  # dy correction for i
    if j in index_to_column:
        col_x, col_y = index_to_column[j]
        B_alpha[m, col_x] = parameters[4]  # dx correction for j
        B_alpha[m, col_y] = parameters[5]  # dy correction for j

B_sides = np.zeros([5, 12], dtype=float)
for n, (i, j) in enumerate(side_pairs):
    # Extract coordinates
    xi, yi = points[i]
    xj, yj = points[j]
    # Calculate parameters
    side_parameters = calculate_side_design_matrix(xi, yi, xj, yj)
    # Map corrections for k, i, and j if they are in index_to_column
    if i in index_to_column:
        col_x, col_y = index_to_column[i]
        B_sides[n, col_x] = side_parameters[0]  # dx correction for i
        B_sides[n, col_y] = side_parameters[1]  # dy correction for i
    if j in index_to_column:
        col_x, col_y = index_to_column[j]
        B_sides[n, col_x] = side_parameters[2]  # dx correction for j
        B_sides[n, col_y] = side_parameters[3]  # dy correction for j

B = np.concatenate((B_alpha, B_sides), axis=0)


# least square and renew all the parameters
N = B.T @ P @ B # B^T P B
N_inv = np.linalg.pinv(N)
C = B.T @ P @ Y
x_new = N_inv @ C
e = Y - B @ x_new
# print("Corrected Xs:")
print(x_new)

# calculate the variance
D = sigma0**2 * N_inv
sig_m = e.T @ P @ e
sigma0 = np.sqrt(sig_m/(20-9))
print(f"sigma_0 = {sigma0}")

# Correct the coordinates of points 0 to 5
for point_index, (col_x, col_y) in index_to_column.items():
    # Ensure x is array-like and access its elements
    dx = x_new[col_x]  # Correction for x
    dy = x_new[col_y]  # Correction for y
    x_old, y_old = points[point_index]
    points[point_index] = (x_old + dx, y_old + dy)

# Print the corrected points
print("Corrected points:")
for point_index, coords in points.items():
    print(f"Point {point_index}: {coords}")

