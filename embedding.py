import os
import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet

# Initialize FaceNet
embedder = FaceNet()

dataset_path = "dataset"

known_embeddings = []
known_names = []

# Loop through dataset
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        # Read image
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize for FaceNet
        img = cv2.resize(img, (160, 160))

        # Expand dims (required)
        img = np.expand_dims(img, axis=0)

        # Get embedding
        embedding = embedder.embeddings(img)[0]

        known_embeddings.append(embedding)
        known_names.append(person_name)

# Save embeddings
data = {"embeddings": known_embeddings, "names": known_names}

with open("embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Embeddings saved successfully!")