import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import csv

g = 9.80665 # m/s^2

# Sensor data
acc_mps2 = np.array([])
# acc_mps2 = np.array([[9.81, 0.1, -0.1],
#                      [0.1, 9.70, 0.1],
#                      [-0.1, 9.8, 0.0]])

# Read calibration raw datapoints
file_name = r'C:\Users\kylew\OneDrive\Documents\Programming\Python\IMU_Calibration\imu_cal_data.txt'
with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        np.append(acc_mps2, [row[0], row[1], row[2]])
        line_count += 1
        print(f'#{line_count}: [{row[0]}, {row[1]}, {row[2]}]')
    print(f'Processed {line_count} lines.')

# Initial guesses for LM algorithm
alpha_yz_init = 0.0
alpha_zy_init = 0.0
alpha_zx_init = 0.0
s_x_init = 1.0
s_y_init = 1.0
s_z_init = 1.0
b_x_init = 0.0
b_y_init = 0.0
b_z_init = 0.0

theta_XL_init = np.array([alpha_yz_init, alpha_zy_init, alpha_zx_init,
                          s_x_init, s_y_init, s_z_init,
                          b_x_init, b_y_init, b_z_init])

def cost(theta_XL):
    alpha_yz = theta_XL[0]
    alpha_zy = theta_XL[1]
    alpha_zx = theta_XL[2]
    s_x = theta_XL[3]
    s_y = theta_XL[4]
    s_z = theta_XL[5]
    b_x = theta_XL[6]
    b_y = theta_XL[7]
    b_z = theta_XL[8]
    
    sum = 0
    for data in acc_mps2:
        a_s = np.array([[data[0]],
                        [data[1]],
                        [data[2]]])
        
        # Transform from measurement frame to orthogonal body frame
        T = np.array([[1, -alpha_yz,  alpha_zy],
                      [0,         1, -alpha_zx],
                      [0,         0,         1]])
        
        # XL scale error
        K = np.array([[s_x,   0,   0],
                      [  0, s_y,   0],
                      [  0,   0, s_z]])
        
        b = np.array([[b_x],
                      [b_y],
                      [b_z]])
        
        h = T @ K @ (a_s + b)
        h_norm_squared = h[0]**2 + h[1]**2 + h[2]**2
        sum += (g**2 - h_norm_squared)**2
    
    return sum

def fun(theta_XL):
    return cost(theta_XL) # - cost(theta_XL_init)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
# result = least_squares(fun, theta_XL_init)
# result = opt.least_squares(fun, theta_XL_init, method='lm', ftol=1e-9, xtol=1e-9)
result = opt.curve_fit(fun, theta_XL_init)

print("Success: {}".format(result.success))
print("{}".format(result.message))
print("alpha_yz = %.3f" % (result.x[0]))
print("alpha_zy = %.3f" % (result.x[1]))
print("alpha_zx = %.3f" % (result.x[2]))
print("s_x = %.3f" % (result.x[3]))
print("s_y = %.3f" % (result.x[4]))
print("s_z = %.3f" % (result.x[5]))
print("b_x = %.3f" % (result.x[6]))
print("b_y = %.3f" % (result.x[7]))
print("b_z = %.3f" % (result.x[8]))
