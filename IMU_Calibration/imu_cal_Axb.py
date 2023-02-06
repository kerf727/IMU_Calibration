import numpy as np
import matplotlib.pyplot as plt
import csv

# Source for calibration method
# https://cookierobotics.com/061/

g = 16384.0

# Sensor data
A = np.array([])

''' Read raw calibration datapoints '''

file_name = r'C:\Users\kylew\OneDrive\Documents\Programming\Python\IMU_Calibration\mpu6050_board1_cal_data_6pt.txt'
with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        A = np.append(A, [float(row[0]), float(row[1]), float(row[2])])
        line_count += 1
        # print(f'#{line_count}: [{row[0]}, {row[1]}, {row[2]}]')
    print(f'Processed {line_count} data points.')

''' Perform calibration '''

A = np.reshape(A, (line_count, 3))

b = np.array([[ 1,  0,  0],
              [-1,  0,  0],
              [ 0,  1,  0],
              [ 0, -1,  0],
              [ 0,  0,  1],
              [ 0,  0, -1]])

# A * x = b
# x = (AT * A)^(-1) * AT * b
AT = np.matrix.transpose(A)
temp = np.linalg.inv(AT @ A)
x = temp @ AT @ b

''' Print results '''

# print("A:")
# print(A)

# print("x:")
# print(x)

post_cal = A @ x
print("A*x:")
np.set_printoptions(precision=4, suppress=True)
print(post_cal)

pre_cal = A / g
print("A / g:")
print(pre_cal)

print("Difference from pre-cal to post-cal. Positive means improvement in error")
print(abs(pre_cal) - abs(post_cal))