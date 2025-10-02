# YOLO-inference
This folder contains Python scripts used for running a custom local AI model and managing the dataset collection and training workflow. The three main files included are:

Running command means python3 commandNAME.py


camera_capture.py
Used to capture image data with your camera. These images can later be uploaded to Roboflow for annotation.


post_annotation.py
Handles the importing of annotated data downloaded from Roboflow.
⚠️ Note: You will need to open this script and update the file paths or address to match the format and location provided by your Roboflow export.


train.py
Contains the training logic and configuration parameters such as the number of epochs. This script is used to train your custom AI model using the prepared dataset.

testModel.py
Implement real-time model inference using webcam stream
