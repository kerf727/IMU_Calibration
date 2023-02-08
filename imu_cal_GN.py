import numpy as np
import csv
# import matplotlib.pyplot as plt
# import scipy.optimize as opt
# from scipy.sparse.linalg import lsmr
# from autograd import grad, jacobian

g = 16384.0

np.set_printoptions(suppress=True)

# Sensor data
acc_mps2 = np.array([])

# Read calibration raw datapoints
file_name = r'C:\Users\kylew\OneDrive\Documents\Programming\Python\IMU_Calibration\mpu6050_board1_cal_data.txt'
with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    num_datapts = 0
    for row in csv_reader:
        acc_mps2 = np.append(acc_mps2, [float(row[0]), float(row[1]), float(row[2])])
        num_datapts += 1
        # print(f'#{num_datapts}: [{row[0]}, {row[1]}, {row[2]}]')
    print(f'Processed {num_datapts} data points.')

acc_mps2 = np.reshape(acc_mps2, (num_datapts, 3))
# print(acc_mps2)

# Initial guesses for least squares fit
#  Initialized to approximate scale factors and biases
#  using standard 6-position calibration method
s_x_init      = 0.9948689923
s_y_init      = 0.9986894639
s_z_init      = 0.9945066618
b_x_init      = -6.5
b_y_init      = -67.5
b_z_init      = -531.5
alpha_yz_init = 0.0
alpha_zy_init = 0.0
alpha_zx_init = 0.0

theta_XL = np.array([[s_x_init],
                     [s_y_init],
                     [s_z_init],
                     [b_x_init],
                     [b_y_init],
                     [b_z_init],
                     [alpha_yz_init],
                     [alpha_zy_init],
                     [alpha_zx_init]])
num_params = len(theta_XL)

epsilon = 1e-9
max_iterations = 100

J = np.zeros((num_datapts, num_params))
R = np.zeros((num_datapts, 1))
A = np.zeros((num_params, num_params))
b = np.zeros((num_params, 1))
prev_b = b

change = 1e6 # initialize to anything greater than epsilon
num_iterations = 1
while change > epsilon and num_iterations <= max_iterations:
    print("Iteration {}. Current change: {:.3f}".format(num_iterations, change))
    t0 = theta_XL[0] # s_x
    t1 = theta_XL[1] # s_y
    t2 = theta_XL[2] # s_z
    t3 = theta_XL[3] # b_x
    t4 = theta_XL[4] # b_y
    t5 = theta_XL[5] # b_z
    t6 = theta_XL[6] # alpha_yz
    t7 = theta_XL[7] # alpha_zy
    t8 = theta_XL[8] # alpha_zx
    
    # sum = 0.0
    for i, data in enumerate(acc_mps2):
        ax = data[0]
        ay = data[1]
        az = data[2]
        
        # XL data with bias error included
        ax_b = ax + t3
        ay_b = ay + t4
        az_b = az + t5
        
        # h(theta) sensor error model. h = T * K * (a_s + b)
        aox = t0*ax_b - t6*t1*ay_b + t7*t2*az_b
        aoy =              t1*ay_b - t8*t2*az_b
        aoz =                           t2*az_b
        
        # Residual function
        # TODO: does this need to be abs()?
        R[i] = aox**2 + aoy**2 + aoz**2 - g**2
        
        # Jacobian of residual function
        # TODO: Confirm these formulae again
        J[i][0] =     2*ax_b*aox
        J[i][1] = -2*t6*ay_b*aox +    2*ay_b*aoy
        J[i][2] =  2*t7*az_b*aox - 2*t8*az_b*aoy + 2*aoz**2
        J[i][3] =       2*t0*aox
        J[i][4] =   -2*t1*t6*aox +      2*t1*aoy
        J[i][5] =    2*t7*t2*aox -   2*t8*t2*aoy + 2*t2**2*az_b
        J[i][6] = -2*t1*ay_b*aox
        J[i][7] =      2*aoz*aox
        J[i][8] =                     -2*aoz*aoy

    JT = np.matrix.transpose(J)
    A = JT @ J
    b = -JT @ R
    # JT*R is the gradient of the cost function

    # A * x = b. x = delta
    # delta = (AT * A)^(-1) * AT * b
    AT = np.matrix.transpose(A)
    temp = np.linalg.inv(AT @ A)
    delta = temp @ AT @ b
    
    # update theta parameters. delta = theta_new - theta_old
    theta_XL += delta
    
    # update magnitude of change in gradient
    d = b - prev_b
    change = d[0]*d[0] + d[1]*d[1] + d[2]*d[2] + \
             d[3]*d[3] + d[4]*d[4] + d[5]*d[5] + \
             d[6]*d[6] + d[7]*d[7] + d[8]*d[8]
    change = change[0]**0.5
    
    prev_b = b
    num_iterations += 1

# Display results
num_improved = 0
num_worsened = 0
for i, data in enumerate(acc_mps2):
    ax = data[0]
    az = data[1]
    ay = data[2]
    ax_b = ax + t3
    ay_b = ay + t4
    az_b = az + t5
    aox = t0*ax_b - t6*t1*ay_b + t7*t2*az_b
    aoy =              t1*ay_b - t8*t2*az_b
    aoz =                           t2*az_b
    error_init = abs(ax**2 + ay**2 + az**2 - g**2)
    error_cal = abs(aox**2 + aoy**2 + aoz**2 - g**2)
    ratio = (error_init / error_cal)[0]
    if ratio >= 1.0:
        print("Observation {}: Error improved by {:.3f}x".format(i, ratio))
        num_improved += 1
    else:
        print("        Observation {}: Error worsened by {:.3f}x".format(i, 1 / ratio))
        num_worsened += 1
    # print(aox, aoy, aoz)

print("Number improved: {}".format(num_improved))
print("Number worsened: {}".format(num_worsened))
print("Final theta parameters:\n{}".format(theta_XL))
