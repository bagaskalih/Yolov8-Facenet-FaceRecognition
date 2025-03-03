import cv2
import numpy as np
import os
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Paths and configurations
KNOWN_FACES_DIR = "known_faces"  # Directory for storing known face embeddings
MODEL_PATH = "yolov8n-face.pt"   # Path to YOLOv8-Face model
CONFIDENCE_THRESHOLD = 0.5
EMBEDDING_SIMILARITY_THRESHOLD = 0.7

# Create directory for known faces if it doesn't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

class FaceRecognition:
    def __init__(self):
        # Load YOLOv8-Face model for detection
        self.detector = YOLO(MODEL_PATH)
        
        # Load FaceNet model for recognition
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # For alignment before recognition (optional but improves accuracy)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # Dictionary to store known face embeddings
        self.known_face_embeddings = {}
        self.load_known_faces()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        
    def load_known_faces(self):
        """Load known faces from directory"""
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith('.npy'):
                name = os.path.splitext(filename)[0]
                path = os.path.join(KNOWN_FACES_DIR, filename)
                self.known_face_embeddings[name] = np.load(path)
                print(f"Loaded face embedding for {name}")
    
    def get_embedding(self, face_img):
        """Generate embedding from face image using FaceNet"""
        # Convert BGR to RGB
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img_pil = Image.fromarray(face_img_rgb)
        
        # Detect face and get aligned face
        try:
            boxes, _ = self.mtcnn.detect(face_img_pil)
            if boxes is None:
                return None
                
            # Convert to PyTorch tensor and normalize
            face_tensor = self.mtcnn(face_img_pil)
            if face_tensor is None:
                return None
                
            # If multiple faces detected, use the first one
            if face_tensor.ndim == 4:
                face_tensor = face_tensor[0].unsqueeze(0)
                
            # Get embedding
            with torch.no_grad():
                face_tensor = face_tensor.to(self.device)
                embedding = self.facenet(face_tensor).cpu().numpy()[0]
            
            return embedding
        except Exception as e:
            print(f"Error in face embedding: {e}")
            return None
    
    def register_new_face(self, face_img, name):
        """Register a new face in the database"""
        embedding = self.get_embedding(face_img)
        if embedding is not None:
            np.save(os.path.join(KNOWN_FACES_DIR, f"{name}.npy"), embedding)
            self.known_face_embeddings[name] = embedding
            print(f"Registered new face for {name}")
            return True
        return False
    
    def identify_face(self, embedding):
        """Identify a face embedding by comparing with known faces"""
        if not self.known_face_embeddings or embedding is None:
            return "Unknown", 0.0
        
        max_similarity = 0
        recognized_name = "Unknown"
        
        for name, known_embedding in self.known_face_embeddings.items():
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                recognized_name = name
        
        if max_similarity < EMBEDDING_SIMILARITY_THRESHOLD:
            return "Unknown", max_similarity
        
        return recognized_name, max_similarity
    
    def run(self):
        """Main loop for face detection and recognition"""
        register_mode = False
        new_face_name = ""
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Run YOLOv8-Face detection
            results = self.detector(frame, conf=CONFIDENCE_THRESHOLD)
            
            # Process detected faces
            for result in results:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = float(box.conf[0])
                    
                    # Extract face region with margin
                    h, w = frame.shape[:2]
                    # Add some margin (10%) around the face
                    margin_x = int((x2 - x1) * 0.1)
                    margin_y = int((y2 - y1) * 0.1)
                    face_x1 = max(0, x1 - margin_x)
                    face_y1 = max(0, y1 - margin_y)
                    face_x2 = min(w, x2 + margin_x)
                    face_y2 = min(h, y2 + margin_y)
                    face_img = frame[face_y1:face_y2, face_x1:face_x2]
                    
                    if face_img.size == 0:
                        continue
                    
                    if register_mode:
                        # Draw rectangle in green for registration mode
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, "Registering new face...", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # Get embedding and identify
                        embedding = self.get_embedding(face_img)
                        if embedding is not None:
                            name, similarity = self.identify_face(embedding)
                            
                            # Draw rectangle and name
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{name} ({similarity:.2f})"
                            cv2.putText(display_frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display instructions
            if register_mode:
                cv2.putText(display_frame, f"Press 'c' to capture for {new_face_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'q' to quit registration mode", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Press 'r' to register new face", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(display_frame, "Press 'q' to quit", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow("Face Recognition", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if register_mode:
                    register_mode = False
                    new_face_name = ""
                else:
                    break
            elif key == ord('r') and not register_mode:
                register_mode = True
                new_face_name = input("Enter name for the new face: ")
            elif key == ord('c') and register_mode:
                # Check if a face is detected
                if results and len(results[0].boxes) > 0:
                    # Get the first detected face for simplicity
                    box = results[0].boxes[0].cpu().numpy()  # Convert to numpy first
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    # Add margin to face
                    h, w = frame.shape[:2]
                    margin_x = int((x2 - x1) * 0.1)
                    margin_y = int((y2 - y1) * 0.1)
                    face_x1 = max(0, x1 - margin_x)
                    face_y1 = max(0, y1 - margin_y)
                    face_x2 = min(w, x2 + margin_x)
                    face_y2 = min(h, y2 + margin_y)
                    face_img = frame[face_y1:face_y2, face_x1:face_x2]
                    
                    if face_img.size > 0:
                        if self.register_new_face(face_img, new_face_name):
                            register_mode = False
                            new_face_name = ""
                        else:
                            print("Failed to register face. Please try again.")
                else:
                    print("No face detected for registration")
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if YOLOv8 model exists
    if not os.path.exists(MODEL_PATH):
        print(f"YOLOv8-Face model not found at {MODEL_PATH}")
        print("Please download the model using: pip install ultralytics && yolo download yolov8n-face.pt")
        exit(1)
    
    try:
        face_recognition = FaceRecognition()
        face_recognition.run()
    except Exception as e:
        print(f"Error: {e}")