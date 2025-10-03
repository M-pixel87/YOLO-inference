from ultralytics import YOLO

# Load your custom-trained PyTorch model
model = YOLO('runs/detect/train/weights/best.pt')

# Export the model to TensorRT format
print("Starting TensorRT export... This may take a few minutes.")
model.export(format='engine') 
print("✅ TensorRT export complete! 'best.engine' file created.")
