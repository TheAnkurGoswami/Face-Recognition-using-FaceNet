import cv2
import numpy as np
from numpy.typing import NDArray
import sys
import os

# Add the project root to sys.path to allow importing 'config'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config # Assuming config.py is in the root and accessible

class FaceDetector:
    """
    A class to detect faces in images using a Haar cascade classifier.
    """

    def __init__(self, cascade_path: str = config.CASCADE_PATH):
        """
        Initializes the FaceDetector.

        Args:
            cascade_path: Path to the Haar cascade XML file.
                          Defaults to the path specified in config.py.
        """
        try:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise IOError(f"Failed to load Haar cascade from {cascade_path}")
        except Exception as e:
            # Consider logging the error here
            print(f"Error loading Haar cascade: {e}") # Or raise a custom exception
            # For now, re-raise to indicate critical failure if cascade doesn't load
            raise

    def detect(self, image: NDArray, scale_factor: float = 1.1, min_neighbors: int = 5) -> list[tuple[int, int, int, int]]:
        """
        Detects faces in a given image.

        Args:
            image: The input image (as a NumPy array, BGR format).
            scale_factor: Parameter specifying how much the image size is reduced at each image scale.
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.

        Returns:
            A list of tuples, where each tuple represents a detected face
            as (x, y, width, height). Returns an empty list if no faces are detected.
        """
        if image is None:
            # print("Input image is None.") # Optional: log this
            return []
        if image.ndim != 3 or image.shape[2] != 3:
            # print(f"Warning: Input image should be a 3-channel BGR image. Got shape {image.shape}") # Optional: log this
            # Attempt to convert if it's a valid image but not 3 channels, or handle as error
            # For now, let's assume it must be BGR and proceed, or return empty
            # For robustness, could try to convert grayscale to BGR:
            # if image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # else: return [] # Or raise ValueError
            pass # Assuming conversion or validation happens before this call for now

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

if __name__ == '__main__':
    # Example Usage (Optional: for testing the detector independently)
    # This part will only run if face_detector.py is executed directly.
    # You'd need a sample image to test this.

    # Create a dummy config for testing if config.py is not yet fully integrated
    # or if you want to override paths for a specific test.
    class DummyConfig:
        CASCADE_PATH = './data/cascade/haarcascade_frontalface_default.xml' # Adjust if needed

    try:
        # detector = FaceDetector(cascade_path=DummyConfig.CASCADE_PATH) # Use if testing with dummy
        detector = FaceDetector() # Assumes config.py is in PYTHONPATH or root

        # Load a test image (replace with an actual image path)
        # Ensure the test image path is correct relative to where this script might be run from for testing.
        # For example, if you run `python core/face_detector.py` from the root directory:
        test_image_path = './data/images/Robert Downey Jr..jpg'
        # If running from inside `core` directory, path might be `../data/images/some_image.jpg`

        sample_image = cv2.imread(test_image_path)

        if sample_image is not None:
            print(f"Image loaded successfully from {test_image_path}, shape: {sample_image.shape}")
            detected_faces = detector.detect(sample_image, min_neighbors=10) # Using min_neighbors from image_recognition.py

            if detected_faces:
                print(f"Detected {len(detected_faces)} faces:")
                for i, (x, y, w, h) in enumerate(detected_faces):
                    print(f"  Face {i+1}: x={x}, y={y}, w={w}, h={h}")
                    cv2.rectangle(sample_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display the image (optional)
                # cv2.imshow("Detected Faces", sample_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Save the image with detections (optional)
                # cv2.imwrite("./test/detected_sample.jpg", sample_image)
                # print("Saved image with detections to ./test/detected_sample.jpg")
            else:
                print("No faces detected in the sample image.")
        else:
            print(f"Failed to load sample image from path: {test_image_path}")
            print("Please ensure that the path is correct and the image file exists.")
            print("If you are running this script from the 'core' directory, the path should be relative to 'core', e.g., '../data/images/Robert Downey Jr..jpg'")
            print("If you are running from the root project directory, it should be './data/images/Robert Downey Jr..jpg'")


    except Exception as e:
        print(f"An error occurred during example usage: {e}")
        # print("Make sure you have an image at './data/images/Robert Downey Jr..jpg' or update the path.")
        # print("Also ensure 'config.py' is accessible and 'haarcascade_frontalface_default.xml' is at the path specified in config.py.")
