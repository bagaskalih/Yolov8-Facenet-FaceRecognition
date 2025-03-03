import face_recognition
import numpy as np
import os

KNOWN_FACES_DIR = "dir_foto"  # Folder containing subfolders for each person
ENCODINGS_FILE = "face_encodings.npy"

known_face_encodings = []
known_face_names = []

# Loop through each person's folder
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
    
    if not os.path.isdir(person_folder):
        continue

    encodings = []
    for filename in os.listdir(person_folder):
        file_path = os.path.join(person_folder, filename)
        
        # Load and encode image
        image = face_recognition.load_image_file(file_path)
        encoding = face_recognition.face_encodings(image)
        
        if encoding:
            encodings.append(encoding[0])

    if encodings:
        # Average multiple encodings for stability
        avg_encoding = np.mean(encodings, axis=0)
        known_face_encodings.append(avg_encoding)
        known_face_names.append(person_name)

# Save to file
np.save(ENCODINGS_FILE, {"encodings": known_face_encodings, "names": known_face_names})
print("Face encodings saved successfully!")
