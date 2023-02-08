# IMU_Calibration

- Python script accepting averaged accelerometer data and compute 9 calibration parameters (scale factor, bias, and cross-axis misalignment)
- Uses Gauss-Newton least squares fit to minimize error of calibration parameters
- Plan to expand to gyroscope calibration as well