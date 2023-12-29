import serial
import csv
from datetime import datetime

port = 'COM10'  
baudrate = 9600
timeout = 1

ACTIVITY_NAME = "idle"
data_file_path = "Data/" + ACTIVITY_NAME + "_data.csv"

imu_file = open(data_file_path, "w", newline="")

imu_writer = csv.writer(imu_file)

imu_writer.writerow(["mX", "mY", "mZ", "gX", "gY", "gZ", "aX", "aY", "aZ", "label"])

# Start the serial connection
with serial.Serial(port, baudrate, timeout=timeout) as ser:
    try:
        while True:
            line = ser.readline().decode().strip()
            if not line:
                continue
            data = line.split(",")
            
            if len(data) != 10:
                continue
            
            sensor, mx, my, mz, gx, gy, gz, ax, ay, az = data
            # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if sensor == "READY":
                imu_writer.writerow([mx, my, mz, gx, gy, gz, ax, ay, az, ACTIVITY_NAME])
    except KeyboardInterrupt:
        pass
    finally:
        imu_file.close()
