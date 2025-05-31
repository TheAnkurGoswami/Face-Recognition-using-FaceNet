import sys # To modify path for config import if run directly
import os # To check file existence
# Ensure the root directory is in sys.path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from numpy.typing import NDArray
from keras.models import load_model
import cv2 # For image resizing

import config
from core.utils import prewhiten # Import prewhiten from core.utils

class FaceEmbedder:
    """
    A class to generate face embeddings using a pre-trained Keras model (FaceNet).
    """

    def __init__(self, model_path: str = config.MODEL_PATH):
        """
        Initializes the FaceEmbedder.

        Args:
            model_path: Path to the Keras FaceNet model H5 file.
                        Defaults to the path specified in config.py.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Keras model not found at {model_path}")
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error loading Keras model from {model_path}: {e}")
            # Consider logging the error here
            raise # Re-raise to indicate critical failure

    def embed(self, face_image: NDArray) -> NDArray:
        """
        Generates a 128-dimensional embedding for a given face image.

        The input face image is expected to be a BGR image (NumPy array).
        It will be resized to 160x160, prewhitened, and then fed to the model.

        Args:
            face_image: A single face image as a NumPy array (BGR format).
                        It's assumed this is a cropped face.

        Returns:
            A 128-dimensional NumPy array representing the face embedding.
            Returns a 1D array.
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Input face_image cannot be None or empty.")

        # Resize to model's expected input size (160x160)
        resized_face = cv2.resize(face_image, (160, 160))

        # Prewhiten the image
        prewhitened_face = prewhiten(resized_face)

        # Reshape to (1, 160, 160, 3) for model prediction (batch of 1)
        reshaped_face = prewhitened_face.reshape(-1, 160, 160, 3)

        # Predict to get the embedding
        embedding = self.model.predict(reshaped_face)

        return embedding.flatten() # Flatten to return a 1D array

if __name__ == '__main__':
    # Example Usage
    print("Running FaceEmbedder example...")
    try:
        # This assumes config.py is accessible (due to sys.path modification)
        # and core.utils is also accessible.

        # 1. Initialize FaceDetector to get a face crop
        # (Reusing FaceDetector from the previous step for a complete example)
        from core.face_detector import FaceDetector

        # Check if cascade and model files exist before proceeding with example
        if not os.path.exists(config.CASCADE_PATH):
            print(f"Cascade file not found: {config.CASCADE_PATH}")
            print("Skipping FaceEmbedder example as it depends on FaceDetector.")
        elif not os.path.exists(config.MODEL_PATH):
            print(f"Model file not found: {config.MODEL_PATH}")
            print("Skipping FaceEmbedder example.")
        else:
            print(f"Loading FaceDetector with cascade from: {config.CASCADE_PATH}")
            detector = FaceDetector() # Uses CASCADE_PATH from config

            # 2. Load a sample image
            test_image_path = './data/images/Mark Ruffalo.jpg' # Relative to project root
            if not os.path.exists(test_image_path):
                print(f"Test image not found: {test_image_path}")
                print("Please ensure the image exists to run the example.")
            else:
                print(f"Loading sample image: {test_image_path}")
                sample_image = cv2.imread(test_image_path)

                if sample_image is not None:
                    # 3. Detect faces
                    print("Detecting faces in the sample image...")
                    detected_faces = detector.detect(sample_image, min_neighbors=10)

                    if detected_faces:
                        print(f"Detected {len(detected_faces)} faces.")
                        # Get the first detected face
                        x, y, w, h = detected_faces[0]
                        face_crop = sample_image[y:y+h, x:x+w]
                        print(f"Cropped face from x:{x}, y:{y}, w:{w}, h:{h}")

                        # 4. Initialize FaceEmbedder
                        print(f"Loading FaceEmbedder with model from: {config.MODEL_PATH}")
                        embedder = FaceEmbedder() # Uses MODEL_PATH from config

                        # 5. Generate embedding for the cropped face
                        print("Generating embedding for the cropped face...")
                        embedding = embedder.embed(face_crop)

                        print(f"Generated embedding (shape: {embedding.shape}):")
                        print(embedding[:10]) # Print first 10 values as a sample
                        print("...")
                        print(f"L2 norm of embedding: {np.linalg.norm(embedding)}")


                        # Optional: Display the cropped face
                        # cv2.imshow("Cropped Face", face_crop)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    else:
                        print("No faces detected in the sample image. Cannot generate embedding.")
                else:
                    print(f"Failed to load sample image from {test_image_path}.")

    except FileNotFoundError as fnf_error:
        print(f"File not found error in example: {fnf_error}")
        print("Please ensure all required data/model files are correctly placed and paths in config.py are accurate.")
    except Exception as e:
        print(f"An error occurred during FaceEmbedder example usage: {e}")
        import traceback
        traceback.print_exc()
