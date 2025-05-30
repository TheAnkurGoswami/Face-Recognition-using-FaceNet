import sys # To modify path for config import if run directly
import os # To check file existence
# Ensure the root directory is in sys.path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance
import cv2 # Imported for example usage

import config
from core.utils import l2_normalize # For normalizing query embedding

class FaceRecognizer:
    """
    Recognizes faces by comparing a query embedding against a database of known embeddings.
    """

    def __init__(self, embeddings_path: str = config.EMBEDDINGS_PATH):
        """
        Initializes the FaceRecognizer.

        Args:
            embeddings_path: Path to the .npz file containing known face embeddings and names.
                             Expected to have an array 'a' (embeddings) and 'b' (names).
        """
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")

        try:
            loaded_data = np.load(embeddings_path, allow_pickle=True)
            if 'a' not in loaded_data or 'b' not in loaded_data:
                raise ValueError(f"Embeddings file {embeddings_path} must contain 'a' (embeddings) and 'b' (names) arrays.")

            self.known_embeddings: NDArray = loaded_data['a']
            self.known_names: NDArray = loaded_data['b']

            if self.known_embeddings.ndim != 2 or self.known_embeddings.shape[1] != 128:
                raise ValueError("Known embeddings must be a 2D array with 128 features per embedding.")
            if self.known_names.ndim != 1 or len(self.known_names) != self.known_embeddings.shape[0]:
                raise ValueError("Names array must be 1D and match the number of known embeddings.")

        except Exception as e:
            print(f"Error loading embeddings or names from {embeddings_path}: {e}")
            raise # Re-raise to indicate critical failure

    def recognize(self, query_embedding: NDArray, threshold: float = 1.0) -> tuple[str, float]:
        """
        Recognizes a face by finding the closest match in the known embeddings database.

        Args:
            query_embedding: The 128-dimensional embedding of the face to recognize.
                             This embedding will be L2 normalized by this function.
            threshold: The maximum distance for a match to be considered valid.
                       If the minimum distance is above this threshold, the face is
                       considered "Unidentified".

        Returns:
            A tuple containing:
            - The name of the recognized person (str), or "Unidentified" (str) if no match
              is found below the threshold.
            - The distance (float) to the closest match. If "Unidentified", this is the
              minimum distance found.
        """
        if self.known_embeddings is None or self.known_names is None or self.known_embeddings.size == 0:
            return "Unidentified", float('inf')

        # Ensure query_embedding is L2 normalized
        # Reshape to 2D if it's flat 1D, as l2_normalize expects 1D or 2D (batch)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        normalized_query_embedding = l2_normalize(query_embedding)

        # If normalized_query_embedding became 2D (batch of 1), take the first row for distance calculation
        if normalized_query_embedding.ndim == 2 and normalized_query_embedding.shape[0] == 1:
            normalized_query_embedding = normalized_query_embedding[0]


        distances = []
        for i in range(len(self.known_embeddings)):
            # Known embeddings are assumed to be pre-normalized (as per generate_data.py)
            # Ensure known_embeddings[i] is also treated as a 1D vector for euclidean distance
            known_emb_vector = self.known_embeddings[i]
            if known_emb_vector.ndim > 1: # Should ideally be 1D already
                 known_emb_vector = known_emb_vector.flatten()

            dist = distance.euclidean(normalized_query_embedding, known_emb_vector)
            distances.append(dist)

        if not distances:
            return "Unidentified", float('inf')

        distances_np = np.array(distances) # Use a different variable name to avoid confusion
        min_dist_idx = np.argmin(distances_np)
        min_dist = distances_np[min_dist_idx]

        if min_dist > threshold:
            return "Unidentified", min_dist
        else:
            # Ensure known_names[min_dist_idx] is a string
            name = str(self.known_names[min_dist_idx])
            return name, min_dist

if __name__ == '__main__':
    print("Running FaceRecognizer example...")
    try:
        # Check if necessary files exist
        if not os.path.exists(config.EMBEDDINGS_PATH):
            print(f"Embeddings file not found: {config.EMBEDDINGS_PATH}")
            print("Skipping FaceRecognizer example. Please generate embeddings first (e.g., using a script like the original generate_data.py).")
        elif not os.path.exists(config.MODEL_PATH): # Needed for FaceEmbedder
            print(f"Model file not found: {config.MODEL_PATH}")
            print("Skipping FaceRecognizer example as it depends on FaceEmbedder.")
        elif not os.path.exists(config.CASCADE_PATH): # Needed for FaceDetector
            print(f"Cascade file not found: {config.CASCADE_PATH}")
            print("Skipping FaceRecognizer example as it depends on FaceDetector.")
        else:
            print(f"Loading recognizer with embeddings from: {config.EMBEDDINGS_PATH}")
            recognizer = FaceRecognizer()

            from core.face_detector import FaceDetector
            from core.face_embedder import FaceEmbedder

            print(f"Loading FaceDetector with cascade from: {config.CASCADE_PATH}")
            detector = FaceDetector()
            print(f"Loading FaceEmbedder with model from: {config.MODEL_PATH}")
            embedder = FaceEmbedder()

            test_image_path = './data/images/Scarlett Johansson .jpg'
            if not os.path.exists(test_image_path):
                print(f"Test image not found: {test_image_path}")
                print("Please ensure this image exists for the example, or change the path.")
            else:
                print(f"Loading sample image for recognition: {test_image_path}")
                sample_image = cv2.imread(test_image_path)

                if sample_image is not None:
                    print("Detecting faces...")
                    detected_faces = detector.detect(sample_image, min_neighbors=10)

                    if detected_faces:
                        x, y, w, h = detected_faces[0]
                        face_crop = sample_image[y:y+h, x:x+w]
                        print("Generating embedding for the test face...")
                        query_emb = embedder.embed(face_crop) # Returns 1D embedding

                        print(f"Recognizing face with embedding (L2 norm before internal normalization: {np.linalg.norm(query_emb):.4f})...")
                        name, dist = recognizer.recognize(query_emb, threshold=1.0) # recognize expects 1D or 2D

                        print(f"Recognition result: Name = {name}, Distance = {dist:.4f}")

                        if name != "Unidentified":
                            print(f"Successfully recognized {name} from the test image.")
                        else:
                            print(f"Could not identify the person, or they are not in the known embeddings. Min distance: {dist:.4f}")
                    else:
                        print("No faces detected in the test image.")
                else:
                    print(f"Failed to load test image from {test_image_path}.")

    except FileNotFoundError as fnf_error:
        print(f"File not found error in example: {fnf_error}")
        print("Ensure all required data/model/embedding files are correctly placed and paths in config.py are accurate.")
        print("You might need to run a data generation script first to create 'embeddings.npz'.")
    except Exception as e:
        print(f"An error occurred during FaceRecognizer example usage: {e}")
        import traceback
        traceback.print_exc()
