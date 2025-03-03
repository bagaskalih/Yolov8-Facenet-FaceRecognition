import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import argparse

# Configuration
KNOWN_FACES_DIR = "known_faces"  # Directory for storing known face embeddings
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

class FaceRegistrar:
    def __init__(self):
        # Set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load FaceNet model for embedding generation
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # For face detection and alignment
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # Ensure output directory exists
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    
    def get_embedding(self, image_path):
        """Extract face embedding from an image file"""
        try:
            # Read image
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    print(f"File not found: {image_path}")
                    return None
                
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Failed to read image: {image_path}")
                    return None
            else:
                img = image_path  # Already a numpy array
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Detect and align face
            face_tensor = self.mtcnn(img_pil)
            
            if face_tensor is None:
                print(f"No face detected in image")
                return None
            
            # If multiple faces detected, use the largest one
            if face_tensor.ndim == 4 and face_tensor.size(0) > 1:
                # Calculate face areas
                areas = []
                for i in range(face_tensor.size(0)):
                    face = face_tensor[i]
                    # Approximate face area by non-zero pixels
                    area = torch.sum(torch.sum(face > 0, dim=1), dim=1).item()
                    areas.append(area)
                
                # Get the face with largest area
                largest_face_idx = np.argmax(areas)
                face_tensor = face_tensor[largest_face_idx].unsqueeze(0)
                print(f"Multiple faces detected, using the largest one")
            
            # Generate embedding
            with torch.no_grad():
                face_tensor = face_tensor.to(self.device)
                embedding = self.facenet(face_tensor).cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def register_face(self, image_path, name):
        """Register a face from an image file"""
        # Check if face is already registered
        output_path = os.path.join(KNOWN_FACES_DIR, f"{name}.npy")
        if os.path.exists(output_path):
            print(f"✓ Skipping {name} - already registered")
            return True
            
        print(f"Processing {name} from {image_path}")
        
        embedding = self.get_embedding(image_path)
        
        if embedding is not None:
            # Save embedding
            np.save(output_path, embedding)
            print(f"✓ Successfully registered {name}")
            return True
        else:
            print(f"✗ Failed to register {name}")
            return False
    
    def process_directory(self, directory_path):
        """Process all images in a directory"""
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return
        
        success_count = 0
        fail_count = 0
        skip_count = 0
        already_registered_count = 0
        
        print(f"\nProcessing images from {directory_path}...")
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Check if file extension is supported
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in SUPPORTED_FORMATS:
                print(f"Skipping unsupported file: {filename}")
                skip_count += 1
                continue
            
            # Extract name from filename
            name = os.path.splitext(filename)[0]
            
            # Check if already registered
            if os.path.exists(os.path.join(KNOWN_FACES_DIR, f"{name}.npy")):
                print(f"✓ Skipping {name} - already registered")
                already_registered_count += 1
                continue
            
            # Register face
            result = self.register_face(file_path, name)
            if result:
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\nRegistration Summary:")
        print(f"- Successfully registered: {success_count}")
        print(f"- Already registered (skipped): {already_registered_count}")
        print(f"- Failed to register: {fail_count}")
        print(f"- Skipped unsupported files: {skip_count}")
        print(f"- Total files processed: {success_count + fail_count + skip_count + already_registered_count}")
        print(f"- Embeddings saved to: {os.path.abspath(KNOWN_FACES_DIR)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Register faces from images in a directory')
    parser.add_argument('directory', type=str, help='Directory containing face images')
    parser.add_argument('--output', type=str, default=KNOWN_FACES_DIR, 
                       help=f'Output directory for face embeddings (default: {KNOWN_FACES_DIR})')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-registration of faces even if they already exist')
    args = parser.parse_args()
    
    # Update output directory if specified
    global KNOWN_FACES_DIR
    KNOWN_FACES_DIR = args.output
    
    # Create face registrar
    registrar = FaceRegistrar()
    
    # Process directory
    registrar.process_directory(args.directory)

if __name__ == "__main__":
    # Set torch threads for optimal performance
    max_workers = max(1, torch.get_num_threads() - 1)
    torch.set_num_threads(max_workers)
    
    main()