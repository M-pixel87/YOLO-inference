from ultralytics import YOLO

# --- Configuration ---
# Path to your dataset's YAML file (find this in the folder Roboflow created)
DATASET_YAML_PATH = 'Yolo_inferencing-1/data.yaml' 
EPOCHS = 50          # How many times to go through the data. Start with 50-100.
IMAGE_SIZE = 640     # The size of images to train on. 640 is a good default.
# ---------------------

# Load a pretrained model (yolov8n.pt is small and fast)
model = YOLO('yolov8n.pt')

# Train the model
if __name__ == '__main__':
    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=0  # Use GPU 0
    )

print("✅ Training complete! Model saved in the 'runs' directory.")
