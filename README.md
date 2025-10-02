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










TUTORIAL: Using YOLO as the Inferencing Model 

 

PREREQUISITES 

 

-> This guide is specifically for JetPack 5. Please verify your version. -> This tutorial was developed on a Jetson AGX Orin. Performance on a Jetson Nano may be slower. -> A camera must be connected and ready for use. 

1. SETUP PYTORCH 

 

NOTE: If model training fails later, an incorrect PyTorch installation is a likely cause. 

Start in your DIR of choice: git clone git@github.com:M-pixel87/YOLO-inference.git cd YOLO-inference 

-> First, remove any existing PyTorch installations to avoid conflicts: sudo pip3 uninstall torch torchvision -y 

-> Next, install the correct PyTorch wheel for JetPack 5: wget https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl 

-> Then, install torchvision from source: sudo apt-get install libjpeg-dev zlib1g-dev -y git clone --branch v0.15.2 https://github.com/pytorch/vision torchvision cd torchvision export BUILD_VERSION=0.15.2 python3 setup.py install --user cd .. 

-> Finally, verify that CUDA is available to PyTorch: python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 

2. CLONE THE INFERENCE REPO 

-> Clone the project repository and navigate into the new directory. git clone git@github.com:M-pixel87/YOLO-inference.git cd YOLO-inference 

IMPORTANT: Stay in this 'YOLO-inference' directory for all subsequent steps. 

3. INSTALL ULTRALYTICS YOLOv8 

 

-> Install the Ultralytics package using pip: pip3 install ultralytics 

-> Run a quick test to confirm the YOLOv8 installation is working: yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg' 

4. DATA COLLECTION 

 

-> Run the script to capture images from your camera for your dataset. python3 cameraCapture.py 

5. ROBOFLOW SETUP & ANNOTATION 

 

-> Go to https://roboflow.com and create a free account. 

-> Create a new project (e.g., "MyJetsonProject"). -> Set the project type to "Object Detection". 

-> Upload the images you captured in the previous step. 

-> Annotate each image: 

    Create your class names (e.g., "widget", "red_button"). 

    Draw bounding boxes around the objects you want to detect. 

6. GENERATE & EXPORT DATASET FROM ROBOFLOW 

 

-> When you're finished annotating, generate a new version of your dataset. 

-> Export the dataset with the following settings: -> Format: YOLOv8 

-> Roboflow will provide a code snippet for downloading. It will look similar to this: 

from roboflow import Roboflow 

rf = Roboflow(api_key="YOUR_PRIVATE_API_KEY") 

project = rf.workspace().project("your-project-name") 

dataset = project.version(1).download("yolov8") 

7. DOWNLOAD ANNOTATED DATA 

 

-> Use the provided script to download the dataset you just created on Roboflow. python3 postAnotation.py 

8. TRAIN YOUR CUSTOM MODEL 

 

-> Now, run the training script. This will use your annotated dataset to train a custom YOLOv8 model. python3 train.py 

9. TEST THE FINAL MODEL 

 

-> With the camera connected, test your newly trained custom model in real-time. python3 testModel.py 

 
  

Test YOLO install: yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg' 

    DATA COLLECTION 

 

Clone again if not done: git clone git@github.com:M-pixel87/YOLO-inference.git 

Run camera capture: python3 cameraCapture.py 

    ROBOFLOW SETUP 

 

    Go to https://roboflow.com and create a free account. 

    Create a new project (e.g., "MyJetsonProject") 

    Type: Object Detection 

    Upload your captured images. 

    Annotate each image: 

    Create class names (e.g., "widget", "red_button") 

    Draw boxes around objects. 

 

    GENERATE & EXPORT DATASET: 

 

    Format: YOLOv8 

Example download code: 

from roboflow import Roboflow 
rf = Roboflow(api_key="YOUR_PRIVATE_API_KEY") 
project = rf.workspace().project("your-project-name") 
dataset = project.version(1).download("yolov8") 
 

 

    POST-ANNOTATION STEP 

 

Download the annotated dataset locally: 
 python3 postAnotation.py 

    TRAINING 

 

Start training your custom model: 
 python3 train.py 

    TEST THE MODEL 

 

Test the trained model (camera must be connected): 
 python3 testModel.py 

 

NOTE: All commands assume you are in the YOLO-inference folder. 
