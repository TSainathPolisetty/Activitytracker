# Wearable Activity Tracker: Arduino BLE & Edge Impulse
## BMI/CEN 598: Embedded Machine Learning Fall 2023

### Abstract
This project introduces a user-driven approach to health monitoring using a dense neural network model on the Arduino Nano 33 BLE Sense. It allows users to train the model on their unique activities, enhancing personalization and accuracy in tracking a diverse range of exercises.

### Introduction
We aim to shift from generic activity trackers to personalized devices that adapt to individual lifestyles. Using the Arduino Nano 33 BLE Sense's Inertial Measurement Unit (IMU), we focus on motion-based data analysis while prioritizing user privacy.

### Key Highlights
- User-driven training pipeline for personalized activity data collection and model training.
- Simple, intuitive interface for all users.
- Capability to add and recognize new activities for a tailored experience.

### System Design
- **Embedded System Architecture**: Built on Arduino Nano 33 BLE Sense, with states managed by a state machine for data handling and TensorFlow Lite model inference.
- **Machine Learning Model Architecture**: A sequence of dense and dropout layers in TensorFlow, optimized for multi-dimensional datasets and configurable for new activities.

## Demonstration - Click on the image to play video
[![Wearable Activity Tracker Using Arduino nano 33 BLE Sense and Edge impulse Demonstration](https://img.youtube.com/vi/zldZvTIZ98E/maxresdefault.jpg)](https://youtu.be/zldZvTIZ98E)


### Evaluation Approach
- **Metrics Used**: Accuracy, F-score, model size, and power consumption.
- **Performance**: Achieved high accuracy and F-score, confirming the model's robustness and efficiency.

### Results
- **Training and Validation**: High accuracy and F-score, demonstrating minimal overfitting.
- **Real-time Execution**: Efficient inference time, suitable for immediate activity recognition.
- **Comparative Analysis**: Superior computational efficiency and reduced power consumption.

### Conclusion
Our system represents a significant advancement in wearable fitness technology, emphasizing user autonomy, privacy, and precise health data monitoring. Future work will focus on expanding capabilities and refining the user experience.

### References
- Arduino Nano 33 BLE Sense Datasheet
- Relevant research papers and publications

