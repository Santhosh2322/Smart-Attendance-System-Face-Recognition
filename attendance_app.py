import cv2
import numpy as np
import pickle
from datetime import datetime
from keras_facenet import FaceNet

# Load FaceNet
embedder = FaceNet()

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = data["embeddings"]
known_names = data["names"]

# Attendance file
attendance_file = "attendance.csv"

def mark_attendance(name):
    with open(attendance_file, "a") as f:
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        f.write(f"{name},{time_str}\n")

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to exit")

marked_names = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect + embed
    faces = embedder.extract(rgb, threshold=0.95)

    for face in faces:
        embedding = face['embedding']
        x, y, w, h = face['box']

        # Compare with known embeddings
        distances = np.linalg.norm(known_embeddings - embedding, axis=1)
        min_dist = np.min(distances)
        index = np.argmin(distances)

        name = "Unknown"

        if min_dist < 0.9:   # threshold
            name = known_names[index]

            if name not in marked_names:
                mark_attendance(name)
                marked_names.add(name)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()