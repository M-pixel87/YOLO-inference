import cv2
from ultralytics import YOLO

# --- Configuration ---
# Path to your optimized TensorRT model
MODEL_PATH = 'runs/detect/train/weights/best.engine' 

# Camera index (0 is usually the default webcam)
CAMERA_INDEX = 0
# ---------------------

# Load the optimized TensorRT model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the .engine file exists and was generated on this machine.")
    exit()

# Initialize the camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Running live inference with optimized model...")
print("Press 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    # Get the width of the camera frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Run inference on the current frame
    results = model(frame, verbose=False) # Set verbose=False to clean up terminal output

    # Process the results
    for result in results:
        # Iterate through each detected object in the frame
        for box in result.boxes:
            # --- 1. Get Bounding Box Coordinates ---
            # Get the coordinates in (left, top, right, bottom) format
            x1, y1, x2, y2 = box.xyxy[0]
            
            # --- 2. Get Confidence Score (Accuracy) ---
            confidence = box.conf[0]
            
            # --- 3. Get Box Width ---
            box_width = x2 - x1
            
            # --- 4. Determine Horizontal Position ---
            # Calculate the center of the box
            center_x = (x1 + x2) / 2
            
            position = ""
            if center_x < frame_width / 3:
                position = "Left"
            elif center_x < frame_width * (2/3):
                position = "Middle"
            else:
                position = "Right"

            # Print the extracted information to the terminal
            print(f"Position: {position}, "
                  f"Accuracy: {confidence:.2f} ({int(confidence*100)}%), "
                  f"Box Width: {int(box_width)} pixels")

    # Get the annotated frame with bounding boxes drawn on it
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("Optimized Model - Live", annotated_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
