import cv2
from ultralytics import YOLO
import time
import threading
from queue import Queue

# --- Configuration ---
MODEL_PATH = 'runs/detect/train/weights/best.engine' 
CAMERA_INDEX = 0
# ---------------------

# A thread-safe class to handle camera reading
class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"Error: Could not open camera {src}.")
            raise IOError
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        # Use a queue to store frames, with a max size of 1 to always get the latest frame
        self.q = Queue(maxsize=1) 

    def start(self):
        # Start the thread to read frames from the video stream
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                return
            (self.grabbed, frame) = self.stream.read()
            # If the queue is full, the oldest frame is dropped, and the new one is added.
            if not self.q.full():
                self.q.put(frame)

    def read(self):
        # Return the latest frame from the queue
        return self.q.get()

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

# --- Main Program ---

# Load the model
model = YOLO(MODEL_PATH)

# Initialize and start the camera stream thread
print("Starting camera stream...")
cap = CameraStream(src=CAMERA_INDEX).start()
time.sleep(1.0) # Give the camera stream some time to start

print("Running live inference...")
print("Press 'q' to quit.")

# Get frame width once
frame_width = int(cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))

while True:
    # Grab the latest frame from the threaded stream
    frame = cap.read()
    if frame is None:
        break

    # Run inference on the current frame
    results = model(frame, verbose=False) 

    # --- OPTIMIZED POST-PROCESSING ---
    # The results[0].plot() is convenient but slow. For max speed, draw manually.
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]} {confidence:.2f}"
            
            # Draw rectangle and text using OpenCV (faster than .plot())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("High-Performance Inference", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
print("Stopping...")
cap.stop()
cv2.destroyAllWindows()
