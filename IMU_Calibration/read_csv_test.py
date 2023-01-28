import numpy as np
import csv

acc_mps2 = np.array([])

file_name = r'C:\Users\kylew\OneDrive\Documents\Programming\Python\IMU_Calibration\imu_cal_data.txt'
with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        acc_mps2 = np.append(acc_mps2, [float(row[0]), float(row[1]), float(row[2])])
        line_count += 1
        print(f'#{line_count}: [{row[0]}, {row[1]}, {row[2]}]')
    print(f'Processed {line_count} lines.')

acc_mps2 = np.reshape(acc_mps2, (line_count, 3))
print(acc_mps2)

# alpha_yz_init = 0.0
# alpha_zy_init = 0.0
# alpha_zx_init = 0.0
# s_x_init = 1.0
# s_y_init = 1.0
# s_z_init = 1.0
# b_x_init = 0.0
# b_y_init = 0.0
# b_z_init = 0.0

# theta_XL_init = np.array([alpha_yz_init, alpha_zy_init, alpha_zx_init,
#                           s_x_init, s_y_init, s_z_init,
#                           b_x_init, b_y_init, b_z_init])
# x = np.tile(theta_XL_init, (line_count, 1))
# print(x)