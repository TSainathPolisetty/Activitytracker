import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


ACTIVITY_NAME = "idle"
NUM_ACTIVIES = 4
ACCEL_MODEL = True
GYRO_MODEL = False
MAG_MODEL = False

enable_quantization = False
tflite_model_filename = 'tflite_models/activity_tracker.tflite'
model_header_filename = 'BMI_final/model.h'
if enable_quantization:
    tflite_model_filename = 'tflite_models/activity_tracker_quantized.tflite'

def generateTrainTest(data_path):
    df = pd.read_csv(data_path)

    label_mapping = {'walking': 0, 'running': 1, 'lifting': 2, 'idle':3}
    df['label'] = df['label'].replace(label_mapping)

    df = df.sample(frac=1).reset_index(drop=True)

    X = df[['mX', 'mY', 'mZ', 'gX', 'gY', 'gZ', 'aX', 'aY', 'aZ']]  
    y = df['label']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv('Data/train.csv', index=False)
    test_df.to_csv('Data/test.csv', index=False)

def plot_accuracies(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def genTFLiteModel():
    df_train = pd.read_csv('Data/train.csv')
    df_test = pd.read_csv('Data/test.csv')

    if df_train.isnull().values.any() or df_test.isnull().values.any():
        print("NaN values found in the dataset")
    else:
        print("No NaN values in the dataset")

    num_classes = df_train['label'].nunique()
    print(f"Number of classes in the train dataset: {num_classes}")

    if ACCEL_MODEL:
        X_train = df_train.drop(["label", "mX", "mY", "mZ", "gX", "gY", "gZ"], axis=1).values
        y_train = df_train['label'].values
        X_test = df_test.drop(["label", "mX", "mY", "mZ", "gX", "gY", "gZ"], axis=1).values
        y_test = df_test['label'].values
        print(X_train[:5])
    
    elif GYRO_MODEL:
        X_train = df_train.drop(["label", "aX", "aY", "aZ", "gX", "gY", "gZ"], axis=1).values
        y_train = df_train['label'].values
        X_test = df_test.drop(["label", "aX", "aY", "aZ", "gX", "gY", "gZ"], axis=1).values
        y_test = df_test['label'].values
    
    elif MAG_MODEL:
        X_train = df_train.drop(["label", "mX", "mY", "mZ", "aX", "aY", "aZ"], axis=1).values
        y_train = df_train['label'].values
        X_test = df_test.drop(["label", "mX", "mY", "mZ", "aX", "aY", "aZ"], axis=1).values
        y_test = df_test['label'].values

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    imu_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(NUM_ACTIVIES, activation='softmax')
    ])

    imu_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = imu_model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32)
    imu_model.summary()
    plot_accuracies(history)

    loss, accuracy = imu_model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

    predictions = imu_model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    plot_confusion_matrix(y_test, y_pred, classes=['walking', 'running', 'lifting', 'idle'])  # Adjust class names if needed

    # F1 Score calculation
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"--------------------------------------------------------------->F1 Score: {f1}")

    # Convert to TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(imu_model)
    tflite_model = converter.convert()
    with open(tflite_model_filename, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved as {tflite_model_filename}")

def convert_tflite_to_header(input_file, output_file):
    with open(input_file, 'rb') as file:
        model_data = file.read()
    model_array = ', '.join(f'0x{byte:02x}' for byte in model_data)
    with open(output_file, 'w') as file:
        file.write('const unsigned char g_model[] = {\n')
        file.write(model_array)
        file.write('\n};\n')
        file.write('\nconst unsigned int g_model_len = sizeof(g_model);\n')

def getArduinoModel():
    basic_model_size = os.path.getsize(tflite_model_filename)
    print("Model is %d bytes" % basic_model_size)
    convert_tflite_to_header(tflite_model_filename, model_header_filename)
    model_h_size = os.path.getsize(model_header_filename)
    print(f"Header file, model.h, is {model_h_size:,} bytes.")
    print("Arduino Header Created")

if __name__ == '__main__':
    generateTrainTest('Data/ex_data.csv')  # Make sure to provide the correct path to your dataset
    genTFLiteModel()
    getArduinoModel()
