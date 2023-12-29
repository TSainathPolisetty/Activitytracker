#include <Arduino_LSM9DS1.h>

float ax, ay, az;
float gx, gy, gz;
float mx, my, mz;

void setup() {
  Serial.begin(9600);
  
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
}

void loop() {
  // Read Accelerometer
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && IMU.magneticFieldAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    IMU.readMagneticField(mx, my, mz);

    Serial.print("READY,");
    Serial.print(mx);
    Serial.print(",");
    Serial.print(my);
    Serial.print(",");
    Serial.print(mz);
    Serial.print(",");
    Serial.print(gx);
    Serial.print(",");
    Serial.print(gy);
    Serial.print(",");
    Serial.print(gz);
    Serial.print(",");
    Serial.print(ax);
    Serial.print(",");
    Serial.print(ay);
    Serial.print(",");
    Serial.print(az);
    Serial.println();
  }
  // Adjust delay for desired sampling rate
  delay(100);
}
