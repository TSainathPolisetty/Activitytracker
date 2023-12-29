import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, 
    QLineEdit, QLabel, QFileDialog, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer
import csv
import serial
from threading import Thread

class LivePlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowTitle("Activity Tracker")

        self.time_data = np.linspace(0, 70, 100)
        self.num_data_points = len(self.time_data)
        self.acc_data = np.zeros((100, 3))  
        self.gyro_data = np.zeros((100, 3))  
        self.mag_data = np.zeros((100, 3))  

        self.acc_fig, self.acc_canvas = self.create_plot('Accelerometer Data')
        self.gyro_fig, self.gyro_canvas = self.create_plot('Gyroscope Data')
        self.mag_fig, self.mag_canvas = self.create_plot('Magnetometer Data')

        self.layout.addWidget(self.acc_canvas)
        self.layout.addWidget(self.gyro_canvas)
        self.layout.addWidget(self.mag_canvas)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plots)

        self.serial_port = serial.Serial('COM10', 9600, timeout=1)
        self.data_logging_thread = None
        self.is_logging = False
        self.is_inferring = False
        self.inference_thread = None

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.file_path_label = QLabel("Data File Path:", self)
        self.layout.addWidget(self.file_path_label)
        self.file_path_input = QLineEdit(self)
        self.layout.addWidget(self.file_path_input)
        self.file_path_button = QPushButton("Select File", self)
        self.file_path_button.clicked.connect(self.select_file_path)
        self.layout.addWidget(self.file_path_button)

        self.activity_name_label = QLabel("Activity Name:", self)
        self.layout.addWidget(self.activity_name_label)
        self.activity_name_input = QLineEdit(self)
        self.layout.addWidget(self.activity_name_input)

        self.start_stop_button = QPushButton("Start Logging", self)
        self.start_stop_button.clicked.connect(self.toggle_logging)
        self.layout.addWidget(self.start_stop_button)

        self.output_label = QLabel("Inference result will be displayed here.", self)
        self.output_label.setWordWrap(True)  # Optional, for longer text
        self.layout.addWidget(self.output_label)

        self.infer_button = QPushButton("Infer", self)
        self.infer_button.clicked.connect(self.infer_data)
        self.layout.addWidget(self.infer_button)

    def select_file_path(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Select Data File", "", "CSV files (*.csv)")
        if file_name:
            self.file_path_input.setText(file_name)

    def toggle_logging(self):
        if self.is_logging:
            self.is_logging = False
            self.start_stop_button.setText("Start Logging")
            self.timer.stop()
            if self.serial_port.is_open:
                self.serial_port.write(b'x')
                # self.serial_port.close()
            if self.data_logging_thread is not None:
                self.data_logging_thread.join()
                self.data_logging_thread = None
        else:
            self.is_logging = True
            self.start_stop_button.setText("Stop Logging")
            self.timer.start(100)
            try:
                self.serial_port.write(b's')
            except serial.SerialException as e:
                print(f"Error opening serial port: {e}")
                return  
            self.data_logging_thread = Thread(target=self.log_sensor_data, daemon=True)
            self.data_logging_thread.start()

    def create_plot(self, title):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlim(0, self.num_data_points - 1)
        ax.set_ylim(-1, 1)  
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Sensor Reading')
        return fig, canvas

    def update_plots(self):
        # Update Accelerometer plot
        self.acc_fig.clear()
        ax1 = self.acc_fig.add_subplot(111)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            ax1.plot(self.time_data, self.acc_data[:, i], label=f'Acc {axis}')
        ax1.legend()
        self.acc_canvas.draw()

        # Update Gyroscope plot
        self.gyro_fig.clear()
        ax2 = self.gyro_fig.add_subplot(111)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            ax2.plot(self.time_data, self.gyro_data[:, i], label=f'Gyro {axis}')
        ax2.legend()
        self.gyro_canvas.draw()

        # Update Magnetometer plot
        self.mag_fig.clear()
        ax3 = self.mag_fig.add_subplot(111)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            ax3.plot(self.time_data, self.mag_data[:, i], label=f'Mag {axis}')
        ax3.legend()
        self.mag_canvas.draw()
    
    def log_sensor_data(self):
        data_file_path = self.file_path_input.text()
        activity_name = self.activity_name_input.text()
        with open(data_file_path, "a", newline="") as imu_file:
            imu_writer = csv.writer(imu_file)
            imu_writer.writerow(["mX", "mY", "mZ", "gX", "gY", "gZ", "aX", "aY", "aZ", "label"])
            while self.is_logging:
                line = self.serial_port.readline().decode().strip()
                if not line:
                    continue
                data = line.split(",")
                if len(data) != 10:
                    continue
                sensor, mx, my, mz, gx, gy, gz, ax, ay, az = data
                if sensor == "READY":
                    imu_writer.writerow([mx, my, mz, gx, gy, gz, ax, ay, az, activity_name])
                    # Update the plot data
                    self.acc_data = np.roll(self.acc_data, -1, axis=0)
                    self.acc_data[-1, :] = [float(ax), float(ay), float(az)]
                    self.gyro_data = np.roll(self.gyro_data, -1, axis=0)
                    self.gyro_data[-1, :] = [float(gx), float(gy), float(gz)]
                    self.mag_data = np.roll(self.mag_data, -1, axis=0)
                    self.mag_data[-1, :] = [float(mx), float(my), float(mz)]
    
    def infer_data(self):
        if not self.serial_port or not self.serial_port.is_open:
            self.output_label.setText("Serial port not open.")
            return
        if self.is_inferring:
            try:
                self.serial_port.write(b'x')  # Send 'x' to stop inferring
                self.inference_thread.join()  # Wait for the inference thread to finish
                self.inference_thread = None
                self.output_label.setText("Inference stopped.")
                self.infer_button.setText("Start Inferring")
                self.is_inferring = False
            except serial.SerialException as e:
                print(f"Error communicating with Arduino: {e}")
                self.output_label.setText("Error in communication.")
        else:
            try:
                self.serial_port.write(b'i')  # Send 'i' to start inferring
                self.infer_button.setText("Stop Inferring")
                self.is_inferring = True
                # Start a new thread for reading inference results
                self.inference_thread = Thread(target=self.read_inference_results, daemon=True)
                self.inference_thread.start()
            except serial.SerialException as e:
                print(f"Error communicating with Arduino: {e}")
                self.output_label.setText("Error in communication.")
    
    def read_inference_results(self):
        while self.is_inferring:
            if self.serial_port.in_waiting:
                line = self.serial_port.readline().decode().strip()
                self.output_label.setText(f"Inference: {line}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = LivePlotter()
    mainWin.show()
    sys.exit(app.exec_())
