import cv2
import os
import time
import datetime
from deepface import DeepFace
import pandas as pd

# --- Configuration ---
DB_PATH = "database"
RECORDINGS_DIR = "recordings"
# To make a separate folder for recordings if it doesn't exist
if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

# Initialize Camera
cap = cv2.VideoCapture(0) # Use 0 for default camera
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for mp4 can change as needed

# Variables for Logic
is_recording = False
out = None
last_recognition_time = 0
recognition_interval = 2  # Run AI every 2 seconds to save CPU, increase or decrease as needed
current_identity = "Detecting..."

print("--- Security System Active ---")

while True:
    ret, frame = cap.read()
    if not ret: break

    current_time = time.time()
    
    # 1. Periodically Run Recognition
    if current_time - last_recognition_time > recognition_interval:
        try:
            # DeepFace.find looks through the 'database' folder, returns matches
            results = DeepFace.find(img_path=frame, db_path=DB_PATH, enforce_detection=False, silent=True)
            
            if len(results) > 0 and not results[0].empty:
                # Get name from the file path (e.g., database/Person.jpg -> Person)
                full_path = results[0]['identity'][0]
                current_identity = os.path.basename(full_path).split('.')[0]
                
                # If known, stop recording if it was a stranger before
                if is_recording:
                    is_recording = False
                    out.release()
                    print(f"Known person {current_identity} detected. Stopped recording.")
            else:
                current_identity = "STRANGER"
                
                # 2. Start Recording Logic for Stranger
                if not is_recording:
                    is_recording = True
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out = cv2.VideoWriter(f"{RECORDINGS_DIR}/stranger_{timestamp}.mp4", fourcc, 20.0, (frame_width, frame_height))
                    print("⚠️ ALERT: Stranger detected! Recording started.")
            
            last_recognition_time = current_time
        except Exception as e:
            print(f"Error in recognition: {e}")

    # 3. Visuals & Bounding Boxes
    color = (0, 255, 0) if current_identity != "STRANGER" else (0, 0, 255)
    
    # Draw Status Text
    cv2.putText(frame, f"Identity: {current_identity}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if current_identity == "STRANGER":
        cv2.putText(frame, "!!! RECORDING !!!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Record frame
        if out: out.write(frame)

    # Display Feed
    cv2.imshow("Security Monitor", frame)

    if cv2.waitKey(1) == ord('q'): # Quit on 'q', the camera feed window will close
        break

# Cleanup
if out: out.release()
cap.release()
cv2.destroyAllWindows()