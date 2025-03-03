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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Paths and configurations
KNOWN_FACES_DIR = "known_faces"  # Directory for storing known face embeddings
MODEL_PATH = "model/yolov8n-face.pt"   # Path to YOLOv8-Face model
CONFIDENCE_THRESHOLD = 0.5
EMBEDDING_SIMILARITY_THRESHOLD = 0.7

# Performance optimizations
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Use all available cores except one
FRAME_SKIP = 1  # Process every frame (set to 1 to eliminate skipping)
FRAME_RESIZE_FACTOR = 1.0  # Resize input frames (lower = faster processing but less accuracy)
BATCH_PROCESSING = True  # Process multiple faces in batches for faster embedding generation

# Detection smoothing parameters
BOX_HISTORY_SIZE = 3  # Number of previous detections to remember for smoothing
SMOOTH_FACTOR = 0.7   # Weight for current detection vs historical (higher = less smoothing)

# Create directory for known faces if it doesn't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

class DetectionTracker:
    """Track and smooth face detections across frames"""
    def __init__(self, history_size=BOX_HISTORY_SIZE):
        self.tracked_faces = {}  # {face_id: {box_history: [], velocity: [], name: str, similarity: float}}
        self.next_id = 0
        self.history_size = history_size
        self.last_cleanup = time.time()
        self.iou_threshold = 0.3  # Lower threshold to match boxes more aggressively
    
    def update(self, detections):
        """Update tracked faces with new detections"""
        current_time = time.time()
        
        # Process new detections
        matched_ids = set()
        unmatched_detections = []
        
        # Match detections to existing tracked faces
        for x1, y1, x2, y2, name, similarity in detections:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            
            best_match = None
            best_score = float('-inf')
            
            # Find best match among existing tracked faces
            for face_id, face_data in self.tracked_faces.items():
                if face_id in matched_ids:
                    continue
                
                # Get the most recent box
                if not face_data['box_history']:
                    continue
                
                last_box = face_data['box_history'][-1]
                last_x1, last_y1, last_x2, last_y2 = last_box
                
                # Calculate IOU
                iou_score = self._calculate_iou(
                    (x1, y1, x2, y2),
                    (last_x1, last_y1, last_x2, last_y2)
                )
                
                # Calculate predicted position using velocity
                pred_x1, pred_y1, pred_x2, pred_y2 = self._predict_position(face_data)
                pred_center_x = (pred_x1 + pred_x2) / 2
                pred_center_y = (pred_y1 + pred_y2) / 2
                
                # Calculate distance score to predicted position
                distance = ((center_x - pred_center_x) ** 2 + 
                          (center_y - pred_center_y) ** 2) ** 0.5
                distance_score = 1 / (1 + distance/100)  # Normalize distance
                
                # Combine scores
                combined_score = iou_score + distance_score
                
                if combined_score > best_score and (iou_score > self.iou_threshold or distance_score > 0.5):
                    best_score = combined_score
                    best_match = face_id
            
            if best_match is not None:
                # Update existing tracked face
                self._update_track(best_match, (x1, y1, x2, y2), name, similarity, current_time)
                matched_ids.add(best_match)
            else:
                # New face, track later
                unmatched_detections.append((x1, y1, x2, y2, name, similarity))
        
        # Create entries for new detections
        for detection in unmatched_detections:
            self._create_new_track(detection, current_time)
        
        # Clean up old tracks (every 2 seconds)
        if current_time - self.last_cleanup > 0.5:
            self._cleanup_old_tracks(current_time, max_age=0.5)
            self.last_cleanup = current_time
        
        # Return smoothed detections
        return self._get_smoothed_detections()
    
    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        intersection_x1 = max(x1_1, x1_2)
        intersection_y1 = max(y1_1, y1_2)
        intersection_x2 = min(x2_1, x2_2)
        intersection_y2 = min(y2_1, y2_2)
        
        if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
            return 0.0
        
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area
    
    def _predict_position(self, face_data):
        if len(face_data['box_history']) < 2:
            return face_data['box_history'][-1]
        
        # Calculate velocity from last two positions
        curr_box = face_data['box_history'][-1]
        prev_box = face_data['box_history'][-2]
        
        vx1 = curr_box[0] - prev_box[0]
        vy1 = curr_box[1] - prev_box[1]
        vx2 = curr_box[2] - prev_box[2]
        vy2 = curr_box[3] - prev_box[3]
        
        # Predict next position
        pred_x1 = int(curr_box[0] + vx1)
        pred_y1 = int(curr_box[1] + vy1)
        pred_x2 = int(curr_box[2] + vx2)
        pred_y2 = int(curr_box[3] + vy2)
        
        return (pred_x1, pred_y1, pred_x2, pred_y2)
    
    def _update_track(self, face_id, box, name, similarity, current_time):
        face_data = self.tracked_faces[face_id]
        face_data['box_history'].append(box)
        if len(face_data['box_history']) > self.history_size:
            face_data['box_history'] = face_data['box_history'][-self.history_size:]
        if similarity > face_data['similarity']:
            face_data['name'] = name
            face_data['similarity'] = similarity
        face_data['last_seen'] = current_time
    
    def _create_new_track(self, detection, current_time):
        x1, y1, x2, y2, name, similarity = detection
        self.tracked_faces[self.next_id] = {
            'box_history': [(x1, y1, x2, y2)],
            'velocity': [(0, 0, 0, 0)],
            'name': name,
            'similarity': similarity,
            'last_seen': current_time
        }
        self.next_id += 1
    
    def _smooth_box(self, box_history):
        """Apply smoothing to box coordinates"""
        if len(box_history) == 1:
            return box_history[0]
        
        # Take most recent box
        current_box = box_history[-1]
        
        # If we have history, apply smoothing
        if len(box_history) > 1:
            # Calculate average of previous boxes
            avg_x1 = 0
            avg_y1 = 0
            avg_x2 = 0
            avg_y2 = 0
            
            # Weighted average with more weight to recent boxes
            total_weight = 0
            for i, (x1, y1, x2, y2) in enumerate(box_history):
                # Exponential weighting with more weight to recent frames
                weight = (i + 1) / sum(range(1, len(box_history) + 1))
                avg_x1 += x1 * weight
                avg_y1 += y1 * weight
                avg_x2 += x2 * weight
                avg_y2 += y2 * weight
                total_weight += weight
            
            avg_x1 /= total_weight
            avg_y1 /= total_weight
            avg_x2 /= total_weight
            avg_y2 /= total_weight
            
            # Current box with higher weight
            x1, y1, x2, y2 = current_box
            smooth_x1 = int(SMOOTH_FACTOR * x1 + (1 - SMOOTH_FACTOR) * avg_x1)
            smooth_y1 = int(SMOOTH_FACTOR * y1 + (1 - SMOOTH_FACTOR) * avg_y1)
            smooth_x2 = int(SMOOTH_FACTOR * x2 + (1 - SMOOTH_FACTOR) * avg_x2)
            smooth_y2 = int(SMOOTH_FACTOR * y2 + (1 - SMOOTH_FACTOR) * avg_y2)
            
            return (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
        
        return current_box
    
    def _cleanup_old_tracks(self, current_time, max_age=1.0):
        """Remove tracks that haven't been seen for a while"""
        to_remove = []
        for face_id, face_data in self.tracked_faces.items():
            if current_time - face_data['last_seen'] > max_age:
                to_remove.append(face_id)
        
        for face_id in to_remove:
            del self.tracked_faces[face_id]
    
    def _get_smoothed_detections(self):
        smoothed_detections = []
        for face_id, face_data in self.tracked_faces.items():
            if not face_data['box_history']:
                continue
            
            # Apply smoothing to box coordinates
            smooth_box = self._smooth_box(face_data['box_history'])
            name = face_data['name']
            similarity = face_data['similarity']
            
            smoothed_detections.append((*smooth_box, name, similarity))
        
        return smoothed_detections

class FrameProcessor:
    """Process frames in a separate thread"""
    def __init__(self, detector, face_recognizer, max_queue_size=5):
        self.detector = detector
        self.face_recognizer = face_recognizer
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.detection_tracker = DetectionTracker()
        self.last_frame = None
        self.last_results = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_frames)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def add_frame(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()  # Discard oldest frame if queue is full
            except queue.Empty:
                pass
        
        self.frame_count += 1
        if self.frame_count % FRAME_SKIP == 0:  # Only process every nth frame
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
        
        # Store the most recent frame
        self.last_frame = frame
    
    def get_result(self):
        try:
            # Get latest results if available
            while not self.result_queue.empty():
                self.last_results = self.result_queue.get_nowait()
            
            # Return the most recent results with the most recent frame
            if self.last_frame is not None and self.last_results is not None:
                return (self.last_frame, self.last_results)
            return None
        except queue.Empty:
            return None
    
    def _process_frames(self):
        """Main processing loop"""
        last_process_time = time.time()
        while self.running:
            try:
                # Add small delay to prevent excessive CPU usage
                current_time = time.time()
                if current_time - last_process_time < 0.01:  # Limit to ~100 processing attempts per second
                    time.sleep(0.005)
                    continue
                
                last_process_time = current_time
                
                # Get frame with timeout
                frame = self.frame_queue.get(timeout=0.1)
                
                # Resize frame for faster processing if needed
                if FRAME_RESIZE_FACTOR != 1.0:
                    h, w = frame.shape[:2]
                    new_w = int(w * FRAME_RESIZE_FACTOR)
                    new_h = int(h * FRAME_RESIZE_FACTOR)
                    frame_resized = cv2.resize(frame, (new_w, new_h))
                else:
                    frame_resized = frame
                
                # Run YOLOv8-Face detection
                results = self.detector(frame_resized, conf=CONFIDENCE_THRESHOLD)
                
                # Process results
                processed_results = []
                
                # Prepare batch of face images for parallel embedding generation
                face_imgs = []
                face_coords = []
                
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        conf = float(box.conf[0])
                        
                        # Scale coordinates back to original frame size if resized
                        if FRAME_RESIZE_FACTOR != 1.0:
                            scale = 1.0 / FRAME_RESIZE_FACTOR
                            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                        
                        # Extract face region with margin
                        h, w = frame.shape[:2]
                        margin_x = int((x2 - x1) * 0.1)
                        margin_y = int((y2 - y1) * 0.1)
                        face_x1 = max(0, x1 - margin_x)
                        face_y1 = max(0, y1 - margin_y)
                        face_x2 = min(w, x2 + margin_x)
                        face_y2 = min(h, y2 + margin_y)
                        face_img = frame[face_y1:face_y2, face_x1:face_x2]
                        
                        if face_img.size == 0:
                            continue
                        
                        face_imgs.append(face_img)
                        face_coords.append((x1, y1, x2, y2))
                
                # Process face embeddings in parallel (if available)
                identities = []
                if face_imgs:
                    if BATCH_PROCESSING:
                        # Process all faces in one batch
                        embeddings = self.face_recognizer.get_embeddings_batch(face_imgs)
                        
                        # Identify all faces
                        if embeddings:
                            futures = []
                            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                for embedding in embeddings:
                                    if embedding is not None:
                                        future = executor.submit(self.face_recognizer.identify_face, embedding)
                                        futures.append(future)
                                    else:
                                        identities.append(("Unknown", 0.0))
                                
                                for future in futures:
                                    identities.append(future.result())
                        else:
                            identities = [("Unknown", 0.0)] * len(face_imgs)
                    else:
                        # Process each face individually
                        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                            futures = [executor.submit(self._process_single_face, face_img) 
                                    for face_img in face_imgs]
                            
                            for future in futures:
                                identities.append(future.result())
                
                # Combine face coordinates and identities
                for (x1, y1, x2, y2), (name, similarity) in zip(face_coords, identities):
                    processed_results.append((x1, y1, x2, y2, name, similarity))
                
                # Apply tracking and smoothing
                smoothed_results = self.detection_tracker.update(processed_results)
                
                # Put in result queue
                self.result_queue.put(smoothed_results)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame processing: {e}")
    
    def _process_single_face(self, face_img):
        """Process a single face for embedding and identification"""
        embedding = self.face_recognizer.get_embedding(face_img)
        if embedding is not None:
            return self.face_recognizer.identify_face(embedding)
        return "Unknown", 0.0

class FaceRecognizer:
    """Handle face recognition and embedding generation"""
    def __init__(self):
        # Load FaceNet model for recognition
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # For alignment before recognition (optional but improves accuracy)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # Dictionary to store known face embeddings
        self.known_face_embeddings = {}
        self.load_known_faces()
        
        # Optimizations
        self.embedding_cache = {}  # Cache embeddings to avoid recomputing
        self.cache_lock = threading.Lock()
    
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
        # Check cache first (using image hash as key)
        img_hash = hash(face_img.tobytes())
        with self.cache_lock:
            if img_hash in self.embedding_cache:
                return self.embedding_cache[img_hash]
        
        # Convert BGR to RGB
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img_pil = Image.fromarray(face_img_rgb)
        
        # Detect face and get aligned face
        try:
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
            
            # Cache the result
            with self.cache_lock:
                self.embedding_cache[img_hash] = embedding
                
                # Limit cache size
                if len(self.embedding_cache) > 100:
                    # Remove a random key to keep cache size manageable
                    self.embedding_cache.pop(next(iter(self.embedding_cache)))
            
            return embedding
        except Exception as e:
            print(f"Error in face embedding: {e}")
            return None
    
    def get_embeddings_batch(self, face_imgs):
        """Generate embeddings for a batch of face images"""
        if not face_imgs:
            return []
        
        # Convert all images to RGB
        face_imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in face_imgs]
        face_imgs_pil = [Image.fromarray(img_rgb) for img_rgb in face_imgs_rgb]
        
        # Process with MTCNN in batch mode
        try:
            # Create a batch of aligned faces
            aligned_faces = []
            for img_pil in face_imgs_pil:
                face_tensor = self.mtcnn(img_pil)
                if face_tensor is not None:
                    # If multiple faces detected, use the first one
                    if face_tensor.ndim == 4:
                        face_tensor = face_tensor[0].unsqueeze(0)
                    aligned_faces.append(face_tensor)
                else:
                    aligned_faces.append(None)
            
            # Process non-None faces
            valid_faces = [face for face in aligned_faces if face is not None]
            
            if not valid_faces:
                return [None] * len(face_imgs)
            
            # Batch process with FaceNet
            batch_tensor = torch.cat(valid_faces, dim=0).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.facenet(batch_tensor).cpu().numpy()
            
            # Map embeddings back to original order
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
            print(f"Error in batch face embedding: {e}")
            return [None] * len(face_imgs)
    
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

class FaceRecognition:
    def __init__(self):
        # Load YOLOv8-Face model for detection
        self.detector = YOLO(MODEL_PATH)
        
        # Create face recognizer
        self.face_recognizer = FaceRecognizer()
        
        # Create frame processor
        self.frame_processor = FrameProcessor(self.detector, self.face_recognizer)
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        
        # FPS calculation
        self.fps_count = 0
        self.fps_start = time.time()
        self.fps = 0
    
    def run(self):
        """Main loop for face detection and recognition"""
        register_mode = False
        new_face_name = ""
        
        # Start frame processor
        self.frame_processor.start()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Calculate FPS
                self.fps_count += 1
                current_time = time.time()
                if current_time - self.fps_start > 1.0:
                    self.fps = self.fps_count / (current_time - self.fps_start)
                    self.fps_count = 0
                    self.fps_start = current_time
                
                # Display FPS
                cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Add current frame to processor queue
                self.frame_processor.add_frame(frame.copy())
                
                # Get latest results
                latest_result = self.frame_processor.get_result()
                
                if latest_result:
                    _, processed_results = latest_result
                    
                    # Process detected faces
                    for x1, y1, x2, y2, name, similarity in processed_results:
                        if register_mode:
                            # Draw rectangle in green for registration mode
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, "Registering new face...", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
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
                    # Get latest results for registration
                    latest_result = self.frame_processor.get_result()
                    
                    if latest_result and latest_result[1]:
                        # Use the first detected face for registration
                        x1, y1, x2, y2, _, _ = latest_result[1][0]
                        
                        # Extract face region with margin
                        h, w = frame.shape[:2]
                        margin_x = int((x2 - x1) * 0.1)
                        margin_y = int((y2 - y1) * 0.1)
                        face_x1 = max(0, x1 - margin_x)
                        face_y1 = max(0, y1 - margin_y)
                        face_x2 = min(w, x2 + margin_x)
                        face_y2 = min(h, y2 + margin_y)
                        face_img = frame[face_y1:face_y2, face_x1:face_x2]
                        
                        if face_img.size > 0:
                            if self.face_recognizer.register_new_face(face_img, new_face_name):
                                register_mode = False
                                new_face_name = ""
                            else:
                                print("Failed to register face. Please try again.")
                    else:
                        print("No face detected for registration")
        
        finally:
            # Stop frame processor
            self.frame_processor.stop()
            
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set process priority higher (platform-specific)
    try:
        import psutil
        process = psutil.Process(os.getpid())
        process.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    except (ImportError, PermissionError):
        print("Could not set process priority")
    
    # Set torch threads for optimal CPU usage
    torch.set_num_threads(MAX_WORKERS)
    
    # Check if YOLOv8 model exists
    if not os.path.exists(MODEL_PATH):
        print(f"YOLOv8-Face model not found at {MODEL_PATH}")
        print("Please download the model using: pip install ultralytics && yolo download yolov8n-face.pt")
        exit(1)
    
    # Display optimization settings
    print(f"CPU Optimization Settings:")
    print(f"- Using {MAX_WORKERS} CPU cores")
    print(f"- Processing every {FRAME_SKIP} frame(s)")
    print(f"- Frame resize factor: {FRAME_RESIZE_FACTOR}")
    print(f"- Batch processing: {'Enabled' if BATCH_PROCESSING else 'Disabled'}")
    print(f"- Box smoothing: {BOX_HISTORY_SIZE} frames history, {SMOOTH_FACTOR} smoothing factor")
    print(f"- Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    
    try:
        face_recognition = FaceRecognition()
        face_recognition.run()
    except Exception as e:
        print(f"Error: {e}")