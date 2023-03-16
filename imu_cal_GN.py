import numpy as np
import csv
import matplotlib.pyplot as plt

####################################################################
# User Parameters
####################################################################

epsilon = 40
max_iterations = 300
display_error_results = 1
plot_results = 1

####################################################################

g = 16384.0

np.set_printoptions(suppress=True)

####################################################################
# Read raw data
####################################################################

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

####################################################################
# Calibrate XL
####################################################################

# Initial guesses for least squares fit
s_x_init      = 1.0
s_y_init      = 1.0
s_z_init      = 1.0
b_x_init      = 0.0
b_y_init      = 0.0
b_z_init      = 0.0
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

J = np.zeros((num_datapts, num_params))
R = np.zeros((num_datapts, 1))
A = np.zeros((num_params, num_params))
b = np.zeros((num_params, 1))
prev_b = b

change = 1e6 # initialize to anything greater than epsilon
num_iterations = 0
while change > epsilon and num_iterations < max_iterations:
    t0 = theta_XL[0] # s_x
    t1 = theta_XL[1] # s_y
    t2 = theta_XL[2] # s_z
    t3 = theta_XL[3] # b_x
    t4 = theta_XL[4] # b_y
    t5 = theta_XL[5] # b_z
    t6 = theta_XL[6] # alpha_yz
    t7 = theta_XL[7] # alpha_zy
    t8 = theta_XL[8] # alpha_zx
    
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
        R[i] = aox**2 + aoy**2 + aoz**2 - g**2
        
        # Jacobian of residual function
        J[i][0] =     2*ax_b*aox
        J[i][1] = -2*t6*ay_b*aox +    2*ay_b*aoy
        J[i][2] =  2*t7*az_b*aox - 2*t8*az_b*aoy + 2*t2*az_b**2
        J[i][3] =       2*t0*aox
        J[i][4] =   -2*t1*t6*aox +      2*t1*aoy
        J[i][5] =    2*t7*t2*aox -   2*t8*t2*aoy + 2*t2**2*az_b
        J[i][6] = -2*t1*ay_b*aox
        J[i][7] =      2*aoz*aox
        J[i][8] =                     -2*aoz*aoy

    JT = np.matrix.transpose(J)
    delta = -np.linalg.inv(JT @ J) @ (JT @ R)
    
    # update theta parameters. delta = theta_new - theta_old
    theta_XL += delta
    
    # update magnitude of change in gradient
    b = JT @ R
    change = np.linalg.norm(b - prev_b)
    
    prev_b = b
    num_iterations += 1
    
    print("\nIteration {}. Gradient change: {:.3f}. delta magnitude: {:.3e}".format(num_iterations, change, np.linalg.norm(delta)))

####################################################################
# Display results
####################################################################

num_improved = 0
num_worsened = 0
error_init = []
error_cal = []
ratio = []
for i, data in enumerate(acc_mps2):
    i += 1
    t0 = theta_XL[0] # s_x
    t1 = theta_XL[1] # s_y
    t2 = theta_XL[2] # s_z
    t3 = theta_XL[3] # b_x
    t4 = theta_XL[4] # b_y
    t5 = theta_XL[5] # b_z
    t6 = theta_XL[6] # alpha_yz
    t7 = theta_XL[7] # alpha_zy
    t8 = theta_XL[8] # alpha_zx
    
    ax = data[0]
    ay = data[1]
    az = data[2]
    
    ax_b = ax + t3
    ay_b = ay + t4
    az_b = az + t5
    aox = (t0*ax_b - t6*t1*ay_b + t7*t2*az_b)[0]
    aoy =              (t1*ay_b - t8*t2*az_b)[0]
    aoz =                           (t2*az_b)[0]
    
    error_init_i = abs(ax**2 + ay**2 + az**2 - g**2)
    error_cal_i = abs(aox**2 + aoy**2 + aoz**2 - g**2)
    error_init.append(error_init_i)
    error_cal.append(error_cal_i)
    ratio_i = error_init_i / error_cal_i
    ratio.append(ratio_i)
    
    if ratio_i >= 1.0:
        if display_error_results:
            print("Observation {}: Error improved by {:.3f}x".format(i, ratio_i))
        num_improved += 1
    else:
        if display_error_results:
            print("                        Observation {}: Error worsened by {:.3f}x".format(i, 1 / ratio_i))
            print(error_init_i, error_cal_i)
            print(ax, ay, az)
            print(ax_b, ay_b, az_b)
            print(aox, aoy, aoz)
        num_worsened += 1

print()
# print("Number improved: {}".format(num_improved))
# print("Number worsened: {}".format(num_worsened))
print("Ran {} iterations before solution converged.".format(num_iterations))
print()

# Format to paste into MPU6050.c driver file
print("imu->acc_scale[0] = %.7ff;" % (theta_XL[0]))
print("imu->acc_scale[1] = %.7ff;" % (theta_XL[1]))
print("imu->acc_scale[2] = %.7ff;" % (theta_XL[2]))
print("imu->acc_bias[0]  = %.7ff;" % (theta_XL[3]))
print("imu->acc_bias[1]  = %.7ff;" % (theta_XL[4]))
print("imu->acc_bias[2]  = %.7ff;" % (theta_XL[5]))
print("imu->acc_yz_rot   = %.7ff;" % (theta_XL[6]))
print("imu->acc_zy_rot   = %.7ff;" % (theta_XL[7]))
print("imu->acc_zx_rot   = %.7ff;" % (theta_XL[8]))

if (plot_results == 1):
    plt.plot(error_init)
    plt.plot(error_cal)
    # plt.plot(ratio)
    plt.legend(["Uncalibrated", "Calibrated"], loc="upper center")
    plt.title("Error Improvement after Calibration")
    plt.grid(True)
    plt.xlabel("Observation Number")
    plt.ylabel("Error")
    plt.show()
