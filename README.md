______________________________________________________________________                                           
    JETSON AGX ORIN INFERENCE WORKFLOW & CONTROL CENTER
______________________________________________________________________

This folder contains a complete workflow for training and running 
custom YOLOv8 models on the Jetson AGX Orin.

It features a GUI APPLICATION (yoloapplication.py) that manages:
  [+] Data Collection
  [+] Roboflow Integration
  [+] Model Training
  [+] TensorRT Optimization

======================================================================
(!) PREREQUISITES (ONE-TIME SETUP)
======================================================================
TARGET HARDWARE: Jetson AGX Orin (JetPack 5)
HARDWARE REQ:    Camera must be connected.

[STEP 1] SETUP PYTORCH & TORCHVISION
If model training fails later, an incorrect PyTorch install is the 
likely cause. Please run these commands in your terminal:

    1. Uninstall existing torch:
       sudo pip3 uninstall torch torchvision -y

    2. Install PyTorch for JetPack 5:
       wget https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
       pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

    3. Install Torchvision from source:
       sudo apt-get install libjpeg-dev zlib1g-dev -y
       git clone --branch v0.15.2 https://github.com/pytorch/vision torchvision
       cd torchvision
       export BUILD_VERSION=0.15.2
       python3 setup.py install --user
       cd ..

    4. Verify CUDA:
       python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

[STEP 2] INSTALL APPLICATION DEPENDENCIES
You must install the YOLO libraries and the GUI framework.

    pip3 install ultralytics
    pip3 install customtkinter Pillow

======================================================================
(i) HOW TO USE THE CONTROL CENTER
======================================================================
Instead of running scripts manually, use the dashboard.

LAUNCH COMMAND:
    python3 yoloapplication.py

----------------------------------------------------------------------
TAB 1: CAPTURE (Data Collection)
----------------------------------------------------------------------
    1. Navigate to the "Capture" tab.
    2. Click "Launch Camera CaptureScript".
    3. Images are saved locally.
    4. Upload these images to https://roboflow.com and annotate them.

----------------------------------------------------------------------
TAB 2: PULL IMGS (Roboflow Sync)
----------------------------------------------------------------------
    1. In Roboflow, generate a dataset version.
       Make sure the Format is: YOLOv8
    2. Copy the code snippet provided by Roboflow.
    3. In the App "Pull" tab, paste the code into the text box.
    4. Click "Process & Download".
       
       * NOTE: The app will automatically clean the name 
       (e.g., "Yolo_inferencing-10") and copy it to the App Clipboard.

----------------------------------------------------------------------
TAB 3: TRAIN (Model Training)
----------------------------------------------------------------------
    1. DATASET FOLDER: Paste the folder name from the clipboard 
       (e.g., "Yolo_inferencing-10").
    2. MODEL NAME: Type a custom name for this run (e.g., "NormModel").
    3. EPOCHS: Adjust the slider (Start with 50-100).
    4. Click "Start Training".

       * NOTE: Your custom model name is now saved to the clipboard.

----------------------------------------------------------------------
TAB 4: OPTIMIZE (Inference & TensorRT)
----------------------------------------------------------------------
    1. TARGET MODEL NAME: Paste or type your model name 
       (e.g., "NormModel").

    2. BUTTON [1. Test Standard (.pt)]: 
       Verifies the model works using standard PyTorch weights.

    3. BUTTON [2. Create Optimized (.engine)]: 
       Converts the PyTorch model to a TensorRT engine for maximum 
       FPS on the Jetson. 
       * WARNING: This takes a few minutes.

    4. BUTTON [3. Run Live Optimized]: 
       Runs the high-speed inference on your webcam.

======================================================================
FILE STRUCTURE GUIDE
======================================================================
  > yoloapplication.py   - The main Control Center Dashboard.
  > cameraCapture.py     - Raw script for data collection.
  > postAnotation.py     - Download script (Managed by App).
  > train.py             - Training logic (Managed by App).
  > modelOptimizement.py - Exports .pt to .engine (Managed by App).
  > runLiveOptimized.py  - Runs the TensorRT engine (Managed by App).
