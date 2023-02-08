import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import csv

g = 16384.0

# Sensor data
acc_mps2 = np.array([])

# Read calibration raw datapoints
file_name = r'C:\Users\kylew\OneDrive\Documents\Programming\Python\IMU_Calibration\mpu6050_board1_cal_data.txt'
with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        acc_mps2 = np.append(acc_mps2, [float(row[0]), float(row[1]), float(row[2])])
        line_count += 1
        # print(f'#{line_count}: [{row[0]}, {row[1]}, {row[2]}]')
    print(f'Processed {line_count} data points.')

acc_mps2 = np.reshape(acc_mps2, (line_count, 3))
# print(acc_mps2)

# Initial guesses for least squares fit
#  Initialized to approximate scale factors and biases
#  using standard 6-position calibration method
s_x_init = 0.9948689923
s_y_init = 0.9986894639
s_z_init = 0.9945066618
b_x_init = -6.5
b_y_init = -67.5
b_z_init = -531.5
# s_x_init = 1.0
# s_y_init = 1.0
# s_z_init = 1.0
# b_x_init = 0.0
# b_y_init = 0.0
# b_z_init = 0.0
alpha_yz_init = 0.0
alpha_zy_init = 0.0
alpha_zx_init = 0.0

theta_XL_init = [s_x_init, s_y_init, s_z_init,
                 b_x_init, b_y_init, b_z_init,
                 alpha_yz_init, alpha_zy_init, alpha_zx_init]

def cost(theta_XL):
    s_x = theta_XL[0]
    s_y = theta_XL[1]
    s_z = theta_XL[2]
    b_x = theta_XL[3]
    b_y = theta_XL[4]
    b_z = theta_XL[5]
    alpha_yz = theta_XL[6]
    alpha_zy = theta_XL[7]
    alpha_zx = theta_XL[8]
    
    sum = 0.0
    for data in acc_mps2:
        a_s = np.array([[data[0]],
                        [data[1]],
                        [data[2]]])
        
        # Transform from measurement frame to orthogonal body frame
        T = np.array([[1.0, -alpha_yz,  alpha_zy],
                      [0.0,       1.0, -alpha_zx],
                      [0.0,       0.0,       1.0]])
        
        # XL scale error
        K = np.array([[s_x, 0.0, 0.0],
                      [0.0, s_y, 0.0],
                      [0.0, 0.0, s_z]])
        
        b = np.array([[b_x],
                      [b_y],
                      [b_z]])
        
        h = T @ K @ (a_s + b)
        # h = np.matmul(T, np.matmul(K, a_s + b))
        h_norm_squared = h[0]**2 + h[1]**2 + h[2]**2
        sum += (g**2 - h_norm_squared)**2
        # print(h[0], h[1], h[2])
        # print(g, h_norm_squared**0.5)
        # print(h_norm_squared**0.5, s_x, s_y, s_z)
        # print(g**2 - h_norm_squared)
    
    return sum

def fun(theta_XL):
    return cost(theta_XL)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
# result = opt.least_squares(fun, theta_XL_init)
result = opt.least_squares(fun, theta_XL_init, ftol=1e-9, xtol=1e-9)
# result = opt.least_squares(fun, theta_XL_init, method='lm')
# result = opt.least_squares(fun, theta_XL_init, method='lm', ftol=1e-9, xtol=1e-9)
# x = np.tile(theta_XL_init, (line_count, 1))
# result = opt.curve_fit(fun, x, acc_mps2)

print("Success: {}".format(result.success))
print("{}".format(result.message))
print("imu->acc_scale[0] = %.7ff;" % (result.x[0]))
print("imu->acc_scale[1] = %.7ff;" % (result.x[1]))
print("imu->acc_scale[2] = %.7ff;" % (result.x[2]))
print("imu->acc_bias[0]  = %.7ff;" % (result.x[3]))
print("imu->acc_bias[1]  = %.7ff;" % (result.x[4]))
print("imu->acc_bias[2]  = %.7ff;" % (result.x[5]))
print("imu->acc_yz_rot   = %.7ff;" % (result.x[6]))
print("imu->acc_zy_rot   = %.7ff;" % (result.x[7]))
print("imu->acc_zx_rot   = %.7ff;" % (result.x[8]))

print("Initial cost: {}".format(cost(theta_XL_init)))
print("New cost: {}".format(cost(result.x)))
