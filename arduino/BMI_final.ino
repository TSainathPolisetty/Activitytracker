#include <TensorFlowLite.h>
#include <Arduino_LSM9DS1.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

#include "model.h"

int NUM_ACTIVITES = 4;
bool ACCEL_MODEL = true;
bool GYRO_MODEL = false;
bool MAG_MODEL = false;

enum State {
  IDLE,
  LOGGING,
  INFERRING
};

State currentState = IDLE;

// const tflite::Model* model = nullptr;
// tflite::MicroInterpreter* interpreter_activity = nullptr;
// TfLiteTensor* input = nullptr;
// TfLiteTensor* output = nullptr;
// int inference_count = 0;

tflite::MicroErrorReporter tfl_error_reporter;
tflite::ErrorReporter* error_reporter = &tfl_error_reporter;

tflite::AllOpsResolver resolver;
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

float accX = 0, accY = 0, accZ = 0;
float gyroX = 0, gyroY = 0, gyroZ = 0;
float magX = 0, magY = 0, magZ = 0;

void printSensorData(){
  Serial.print("Raw sensor data: ");
  Serial.print(accX);
  Serial.print(',');
  Serial.print(accY);
  Serial.print(',');
  Serial.print(accZ);
  Serial.print(',');
  Serial.print(gyroX);
  Serial.print(',');
  Serial.print(gyroY);
  Serial.print(',');
  Serial.print(gyroZ);
  Serial.print(',');
  Serial.print(magX);
  Serial.print(',');
  Serial.print(magY);
  Serial.print(',');
  Serial.print(magZ);
  Serial.println();
}


void readSensorData(){
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && IMU.magneticFieldAvailable()) {
    IMU.readAcceleration(accX, accY, accZ);
    IMU.readGyroscope(gyroX, gyroY, gyroZ);
    IMU.readMagneticField(magX, magY, magZ);
  }
}

void HandleOutput(float gesture_index) {
  int gesture = static_cast<int>(gesture_index);
  switch (gesture) {
    case 0:
      Serial.println("Activity Detected: Walking");
      break;
    case 1:
      Serial.println("Activity Detected: Running");
      break;
    case 2:
      Serial.println("Activity Detected: Lifting");
      break;
    case 3:
      Serial.println("Activity Detected: Idle");
      break;
    default:
      Serial.println("Activity Detected: Unknown");
  }
}

void logData(){
  readSensorData();
  Serial.print("READY,");
  Serial.print(magX);
  Serial.print(",");
  Serial.print(magY);
  Serial.print(",");
  Serial.print(magZ);
  Serial.print(",");
  Serial.print(gyroX);
  Serial.print(",");
  Serial.print(gyroY);
  Serial.print(",");
  Serial.print(gyroZ);
  Serial.print(",");
  Serial.print(accX);
  Serial.print(",");
  Serial.print(accY);
  Serial.print(",");
  Serial.print(accZ);
  Serial.println();
}

int runInference() {
  const tflite::Model* tfl_model_accel = tflite::GetModel(g_model);
  tflite::MicroInterpreter interpreter_accel(tfl_model_accel, resolver, tensor_arena, kTensorArenaSize);
  interpreter_accel.AllocateTensors();

  TfLiteTensor* input = interpreter_accel.input(0);
  if (input->dims->size != 2 || input->dims->data[0] != 1 || input->dims->data[1] != 3) {
    Serial.println("Unexpected input tensor shape!");
    return -1;
  }
  
  readSensorData();

  // Populate the input tensor
  input->data.f[0] = accX;
  input->data.f[1] = accY;
  input->data.f[2] = accZ;

  Serial.println("Inference Started!");
  if (interpreter_accel.Invoke() != kTfLiteOk) {
    Serial.println("Inference error!");
    return -1;
  }
  // Extract and return the prediction from the output tensor
  TfLiteTensor* output = interpreter_accel.output(0);

  float maxVal = output->data.f[0];
  int predictedClass = 0;
  for (int i = 1; i < output->dims->data[1]; i++) {
    if (output->data.f[i] > maxVal) {
      maxVal = output->data.f[i];
      predictedClass = i;
    }
  } 
  HandleOutput(predictedClass);
}


void setup() {
  // tflite::InitializeTarget();
  // model = tflite::GetModel(g_model);
  // if (model->version() != TFLITE_SCHEMA_VERSION) {
  //   MicroPrintf(
  //       "Model provided is schema version %d not equal "
  //       "to supported version %d.",
  //       model->version(), TFLITE_SCHEMA_VERSION);
  //   return;
  // }

  // static tflite::AllOpsResolver resolver;
  // static tflite::MicroInterpreter static_interpreter(
  //     model, resolver, tensor_arena, kTensorArenaSize);
  // interpreter_activity = &static_interpreter;
  
  // TfLiteStatus allocate_status = interpreter_activity->AllocateTensors();
  // if (allocate_status != kTfLiteOk) {
  //   MicroPrintf("AllocateTensors() failed");
  //   return;
  // }

  Serial.begin(9600);
  while (!Serial);
  Serial.println("Hello World!");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
}

void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();
    switch (receivedChar) {
      case 's':  
        // Start logging
        currentState = LOGGING;
        break;
      case 'i':  
        // Start inferring
        currentState = INFERRING;
        break;
      case 'x':  
        // Stop logging or inferring
        currentState = IDLE;
        break;
    }
  }
  switch (currentState) {
    case LOGGING:
      logData();
      break;
    case INFERRING:
      runInference();
      break;
    case IDLE:
      break;
  }
}

