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
alpha0 = math.degrees(math.atan2(points[1][1], points[1][0])) # Access y as points[1][1] and x as points[1][0]
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
normal_errors = np.random.normal(0,1/3600,15)
alpha_with_errors = alpha + normal_errors
dms_with_errors = [decimal_to_dms(dd) for dd in alpha_with_errors]

# Calculate all the sides
side_pairs = [
    (0, 1), (1, 3), (3, 4), (4, 5), (5, 0)
]
sides = np.zeros(5, dtype=float)
for i, (a,b) in enumerate(side_pairs):
    sides[i] = math.sqrt((points[b][0]-points[a][0])**2+(points[b][1]-points[a][1])**2)
side_normal_errors = np.random.normal(0,0.001,5)
side_with_errors = sides + side_normal_errors





# Part2: Least Square Solutions

# define Q matrix
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



Q = np.block([[Q_alpha, np.zeros((Q_alpha.shape[0], Q_sides.shape[1]))],
                       [np.zeros((Q_sides.shape[0], Q_alpha.shape[1])), Q_sides]])

# Q = np.diag([1] * 20)
P = np.linalg.inv(Q)


# Set input points
points = {
    0: (0, 0),
    1: (5, 2),
    2: (0, 3),
    3: (3, 5),
    4: (-3, 7),
    5: (-5, 2)
}

x_new = np.zeros(9, dtype=float)

# convert observations into decimal degrees
dd_with_errors = [dms_to_decimal(dd) for dd in dms_with_errors]

# start iteration
sigma0 = 1
D = sigma0**2 * Q
i_num = 0



while i_num < 50:
    print(i_num)
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
    Y_alpha = np.radians(Y_alpha)

    sides_corrected = np.zeros(5, dtype=float)
    for i, (a, b) in enumerate(side_pairs):
        sides_corrected[i] = math.sqrt((points[b][0] - points[a][0]) ** 2 + (points[b][1] - points[a][1]) ** 2)
    Y_side = side_with_errors - sides_corrected

    Y = np.concatenate((Y_alpha, Y_side), axis=0)
    #
    index_to_column = {
        2: (1, 2),  # x2, y2
        3: (3, 4),  # x3, y3
        4: (5, 6),  # x4, y4
        5: (7, 8)  # x5, y5
    }

    # set observation matrix
    B_alpha = np.zeros([15, 9], dtype=float)
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
        if k == 1:
            B_alpha[m, 0] = parameters[0]
        if i in index_to_column:
            col_x, col_y = index_to_column[i]
            B_alpha[m, col_x] = parameters[2]  # dx correction for i
            B_alpha[m, col_y] = parameters[3]  # dy correction for i
        if i == 1:
            B_alpha[m, 0] = parameters[2]
        if j in index_to_column:
            col_x, col_y = index_to_column[j]
            B_alpha[m, col_x] = parameters[4]  # dx correction for j
            B_alpha[m, col_y] = parameters[5]  # dy correction for j
        if j == 1:
            B_alpha[m, 0] = parameters[4]



    B_sides = np.zeros([5, 9], dtype=float)
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
        if i == 1:
            B_sides[n, 0] = side_parameters[0]
        if j in index_to_column:
            col_x, col_y = index_to_column[j]
            B_sides[n, col_x] = side_parameters[2]  # dx correction for j
            B_sides[n, col_y] = side_parameters[3]  # dy correction for j
        if j == 1:
            B_sides[n, 0] = side_parameters[2]

    B = np.concatenate((B_alpha, B_sides), axis=0)

    # least square and renew all the parameters
    N = B.T @ P @ B  # B^T P B
    N_inv = np.linalg.inv(N)
    BtPy = B.T @ P @ Y
    x_new = N_inv @ BtPy
    # print("Corrected Xs:")
    # print(x_new)


    # Correct the coordinates of point 1
    x_old, y_old = points[1]
    dx = x_new[0]
    tan0_new = (points[1][1]-points[0][1])/(points[1][0]-points[0][0])
    dy = tanalpha0 * (points[1][0]-points[0][0]) + tan0_new * dx - (points[1][1]-points[0][1])
    points[1] = (x_old + dx, y_old + dy)

    # Correct the coordinates of points 2 to 5
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

    i_num = i_num + 1

    if (np.abs(x_new)).max() < 1e-4:
        print(f"Find the solution!")
        print(f"The number of iterations: {i_num}")
        break