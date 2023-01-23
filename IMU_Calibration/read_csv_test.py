import numpy as np
import csv

file_name = r'C:\Users\kylew\OneDrive\Documents\Programming\Python\IMU_Calibration\imu_cal_data.txt'
acc_mps2 = np.array([])
with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        np.append(acc_mps2, [row[0], row[1], row[2]])
        line_count += 1
        print(f'#{line_count}: [{row[0]}, {row[1]}, {row[2]}]')
    print(f'Processed {line_count} lines.')