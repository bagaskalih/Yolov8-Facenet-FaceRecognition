import cv2
import numpy as np
import os
import time
import threading
import queue
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing

# Paths and configurations
KNOWN_FACES_DIR = "known_faces"  # Directory for storing known face embeddings
MODEL_PATH = "yolov8n-face.pt"   # Path to YOLOv8-Face model
CONFIDENCE_THRESHOLD = 0.5
EMBEDDING_SIMILARITY_THRESHOLD = 0.7

# Performance optimizations
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Use all available cores except one
FRAME_SKIP = 1  # Process every nth frame
FRAME_RESIZE_FACTOR = 1.0  # Resize input frames (lower = faster processing but less accuracy)
BATCH_PROCESSING = True  # Process multiple faces in batches

# Detection smoothing parameters
SMOOTH_FACTOR = 0.7   # Weight for current detection vs historical (higher = less smoothing)
MAX_TRACK_AGE = 0.5   # Maximum time (seconds) to keep a track without updates

# Create directory for known faces if it doesn't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

class SimpleTracker:
    """Simplified tracker for face detections"""
    def __init__(self, iou_threshold=0.3, max_age=MAX_TRACK_AGE):
        self.tracks = {}  # {track_id: {box, name, similarity, last_seen}}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.last_cleanup = time.time()
    
    def update(self, detections):
        """Update tracks with new detections"""
        current_time = time.time()
        matched_tracks = set()
        unmatched_detections = []
        
        # Match detections to existing tracks
        for detection in detections:
            x1, y1, x2, y2, name, similarity = detection
            best_match = None
            best_iou = 0
            
            # Find best match based on IOU
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                tx1, ty1, tx2, ty2 = track['box']
                iou = self._calculate_iou((x1, y1, x2, y2), (tx1, ty1, tx2, ty2))
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = track_id
            
            if best_match is not None:
                # Update existing track
                track = self.tracks[best_match]
                
                # Apply smoothing to box
                tx1, ty1, tx2, ty2 = track['box']
                smooth_x1 = int(SMOOTH_FACTOR * x1 + (1 - SMOOTH_FACTOR) * tx1)
                smooth_y1 = int(SMOOTH_FACTOR * y1 + (1 - SMOOTH_FACTOR) * ty1)
                smooth_x2 = int(SMOOTH_FACTOR * x2 + (1 - SMOOTH_FACTOR) * tx2)
                smooth_y2 = int(SMOOTH_FACTOR * y2 + (1 - SMOOTH_FACTOR) * ty2)
                
                track['box'] = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
                
                # Update name if similarity is higher
                if similarity > track['similarity']:
                    track['name'] = name
                    track['similarity'] = similarity
                
                track['last_seen'] = current_time
                matched_tracks.add(best_match)
            else:
                # New detection
                unmatched_detections.append(detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            x1, y1, x2, y2, name, similarity = detection
            self.tracks[self.next_id] = {
                'box': (x1, y1, x2, y2),
                'name': name,
                'similarity': similarity,
                'last_seen': current_time
            }
            self.next_id += 1
        
        # Clean up old tracks
        if current_time - self.last_cleanup > self.max_age:
            self._cleanup_old_tracks(current_time)
            self.last_cleanup = current_time
        
        # Return current tracks as detections
        return [(track['box'][0], track['box'][1], track['box'][2], track['box'][3], 
                track['name'], track['similarity']) 
                for track in self.tracks.values()]
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def _cleanup_old_tracks(self, current_time):
        """Remove tracks that haven't been seen recently"""
        to_remove = []
        for track_id, track in self.tracks.items():
            if current_time - track['last_seen'] > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        self.last_cleanup = current_time

class FaceRecognitionSystem:
    """Unified face recognition system with simplified pipeline"""
    def __init__(self):
        # Initialize models
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = YOLO(MODEL_PATH)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # Initialize processing components
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue()
        self.tracker = SimpleTracker()
        self.known_faces = {}
        self.embedding_cache = {}
        
        # Processing thread
        self.running = False
        self.process_thread = None
        
        # Frame counter and FPS calculation
        self.frame_count = 0
        self.last_frame = None
        self.last_results = None
        self.fps = 0
        self.fps_count = 0
        self.fps_start = time.time()
        
        # Load known faces
        self.load_known_faces()
        
        # Set torch threads for optimal CPU usage
        torch.set_num_threads(MAX_WORKERS)
    
    def load_known_faces(self):
        """Load known face embeddings from directory"""
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith('.npy'):
                name = os.path.splitext(filename)[0]
                path = os.path.join(KNOWN_FACES_DIR, filename)
                self.known_faces[name] = np.load(path)
                print(f"Loaded face embedding for {name}")
    
    def start_processing(self):
        """Start the processing thread"""
        if not self.running:
            self.running = True
            self.process_thread = threading.Thread(target=self._process_frames)
            self.process_thread.daemon = True
            self.process_thread.start()
    
    def stop_processing(self):
        """Stop the processing thread"""
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
    
    def add_frame(self, frame):
        """Add a frame to the processing queue"""
        self.frame_count += 1
        if self.frame_count % FRAME_SKIP == 0:
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()  # Discard oldest frame if queue is full
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
        
        self.last_frame = frame
    
    def get_result(self):
        """Get the latest processing result"""
        try:
            while not self.result_queue.empty():
                self.last_results = self.result_queue.get_nowait()
            
            if self.last_frame is not None and self.last_results is not None:
                return (self.last_frame, self.last_results)
            return None
        except queue.Empty:
            return None
    
    def register_face(self, frame, face_coords, name):
        """Register a new face in the database"""
        x1, y1, x2, y2 = face_coords
        
        # Extract face with margin
        h, w = frame.shape[:2]
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)
        face_x1 = max(0, x1 - margin_x)
        face_y1 = max(0, y1 - margin_y)
        face_x2 = min(w, x2 + margin_x)
        face_y2 = min(h, y2 + margin_y)
        face_img = frame[face_y1:face_y2, face_x1:face_x2]
        
        if face_img.size == 0:
            return False
        
        # Generate embedding
        embedding = self._generate_embedding(face_img)
        if embedding is None:
            return False
        
        # Save embedding
        np.save(os.path.join(KNOWN_FACES_DIR, f"{name}.npy"), embedding)
        self.known_faces[name] = embedding
        print(f"Registered new face for {name}")
        return True
    
    def _process_frames(self):
        """Main frame processing loop"""
        while self.running:
            try:
                # Get a frame from the queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Resize frame if needed
                if FRAME_RESIZE_FACTOR != 1.0:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(frame, (int(w * FRAME_RESIZE_FACTOR), int(h * FRAME_RESIZE_FACTOR)))
                
                # Run detection
                results = self.detector(frame, conf=CONFIDENCE_THRESHOLD)
                
                # Extract faces
                faces = []
                face_coords = []
                
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        
                        # Scale coordinates back if resized
                        if FRAME_RESIZE_FACTOR != 1.0:
                            scale = 1.0 / FRAME_RESIZE_FACTOR
                            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                        
                        # Extract face with margin
                        h, w = frame.shape[:2]
                        margin_x = int((x2 - x1) * 0.1)
                        margin_y = int((y2 - y1) * 0.1)
                        face_x1 = max(0, x1 - margin_x)
                        face_y1 = max(0, y1 - margin_y)
                        face_x2 = min(w, x2 + margin_x)
                        face_y2 = min(h, y2 + margin_y)
                        face_img = frame[face_y1:face_y2, face_x1:face_x2]
                        
                        if face_img.size > 0:
                            faces.append(face_img)
                            face_coords.append((x1, y1, x2, y2))
                
                # Generate embeddings (batch or individual)
                detections = []
                
                if faces:
                    if BATCH_PROCESSING:
                        # Batch processing of embeddings
                        embeddings = self._generate_embeddings_batch(faces)
                        
                        # Identify faces
                        for (x1, y1, x2, y2), embedding in zip(face_coords, embeddings):
                            if embedding is not None:
                                name, similarity = self._identify_face(embedding)
                                detections.append((x1, y1, x2, y2, name, similarity))
                    else:
                        # Process each face individually
                        for (x1, y1, x2, y2), face_img in zip(face_coords, faces):
                            embedding = self._generate_embedding(face_img)
                            if embedding is not None:
                                name, similarity = self._identify_face(embedding)
                                detections.append((x1, y1, x2, y2, name, similarity))
                
                # Update tracker
                tracked_detections = self.tracker.update(detections)
                
                # Put results in queue
                self.result_queue.put(tracked_detections)
                
            except queue.Empty:
                # No frames available, wait a bit
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in frame processing: {e}")
    
    def _generate_embedding(self, face_img):
        """Generate face embedding for a single face image"""
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # Detect and align face
            face_tensor = self.mtcnn(face_pil)
            if face_tensor is None:
                return None
            
            # If multiple faces detected, use the first one
            if face_tensor.ndim == 4:
                face_tensor = face_tensor[0].unsqueeze(0)
            
            # Generate embedding
            with torch.no_grad():
                face_tensor = face_tensor.to(self.device)
                embedding = self.facenet(face_tensor).cpu().numpy()[0]
            
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def _generate_embeddings_batch(self, face_imgs):
        """Generate embeddings for multiple face images in a batch"""
        try:
            # Convert all images to RGB
            face_imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in face_imgs]
            face_pils = [Image.fromarray(img) for img in face_imgs_rgb]
            
            # Process with MTCNN
            aligned_faces = []
            for img_pil in face_pils:
                face_tensor = self.mtcnn(img_pil)
                if face_tensor is not None:
                    # Handle multi-face detection
                    if face_tensor.ndim == 4:
                        face_tensor = face_tensor[0].unsqueeze(0)
                    aligned_faces.append(face_tensor)
                else:
                    aligned_faces.append(None)
            
            # Gather valid faces
            valid_faces = [face for face in aligned_faces if face is not None]
            if not valid_faces:
                return [None] * len(face_imgs)
            
            # Batch process
            batch_tensor = torch.cat(valid_faces, dim=0).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.facenet(batch_tensor).cpu().numpy()
            
            # Map back to original order
            embeddings = []
            valid_idx = 0
            for face in aligned_faces:
                if face is not None:
                    embeddings.append(batch_embeddings[valid_idx])
                    valid_idx += 1
                else:
                    embeddings.append(None)
            
            return embeddings
        except Exception as e:
            print(f"Error in batch embedding generation: {e}")
            return [None] * len(face_imgs)
    
    def _identify_face(self, embedding):
        """Identify a face by comparing with known face embeddings"""
        if not self.known_faces:
            return "Unknown", 0.0
        
        max_similarity = 0
        recognized_name = "Unknown"
        
        for name, known_embedding in self.known_faces.items():
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                recognized_name = name
        
        if max_similarity < EMBEDDING_SIMILARITY_THRESHOLD:
            return "Unknown", max_similarity
        
        return recognized_name, max_similarity
    
    def run_camera(self):
        """Run face recognition on camera feed"""
        # Start processing thread
        self.start_processing()
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        # Variables for registration mode
        register_mode = False
        new_face_name = ""
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Update FPS counter
                self.fps_count += 1
                current_time = time.time()
                if current_time - self.fps_start > 1.0:
                    self.fps = self.fps_count / (current_time - self.fps_start)
                    self.fps_count = 0
                    self.fps_start = current_time
                
                # Add frame to processing queue
                self.add_frame(frame.copy())
                
                # Get latest results
                latest_result = self.get_result()
                
                if latest_result:
                    _, detections = latest_result
                    
                    # Display detections
                    for x1, y1, x2, y2, name, similarity in detections:
                        if register_mode:
                            # Green box for registration mode
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, "Registering new face...", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            # Regular display
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{name} ({similarity:.2f})"
                            cv2.putText(display_frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display FPS
                cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
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
                    # Register face
                    latest_result = self.get_result()
                    if latest_result and latest_result[1]:
                        x1, y1, x2, y2, _, _ = latest_result[1][0]  # Use first detected face
                        if self.register_face(frame, (x1, y1, x2, y2), new_face_name):
                            register_mode = False
                            new_face_name = ""
                        else:
                            print("Failed to register face. Please try again.")
                    else:
                        print("No face detected for registration")
        
        finally:
            # Clean up
            self.stop_processing()
            cap.release()
            cv2.destroyAllWindows()

def main():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"YOLOv8-Face model not found at {MODEL_PATH}")
        print("Please download the model using: pip install ultralytics && yolo download yolov8n-face.pt")
        exit(1)
    
    # Set process priority if possible
    try:
        import psutil
        process = psutil.Process(os.getpid())
        process.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    except (ImportError, PermissionError):
        print("Could not set process priority")
    
    # Display settings
    print(f"CPU Optimization Settings:")
    print(f"- Using {MAX_WORKERS} CPU cores")
    print(f"- Processing every {FRAME_SKIP} frame(s)")
    print(f"- Frame resize factor: {FRAME_RESIZE_FACTOR}")
    print(f"- Batch processing: {'Enabled' if BATCH_PROCESSING else 'Disabled'}")
    print(f"- Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    
    # Run the system
    system = FaceRecognitionSystem()
    system.run_camera()

if __name__ == "__main__":
    main()