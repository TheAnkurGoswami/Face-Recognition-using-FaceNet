import sys
import os
import numpy as np
import cv2
from numpy.typing import NDArray

# Ensure the root directory is in sys.path for config and core imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config
from core.face_detector import FaceDetector
from core.face_embedder import FaceEmbedder
from core.face_recognizer import FaceRecognizer
from core.drawing_utils import draw_face_rectangle, draw_name_label

# Global default values for font calculation if name is not found in vars
# (Copied from refactored image_recognition.py)
DEFAULT_SLOPE = 0.1974311
DEFAULT_INTERCEPT = 0.03397702412218706

def get_dynamic_font_size(
    name: str,
    face_width_for_font_calc: int,
    known_names_for_font: list[str],
    slopes: NDArray,
    intercepts: NDArray,
    default_slope: float = DEFAULT_SLOPE,
    default_intercept: float = DEFAULT_INTERCEPT
) -> int:
    """
    Calculates dynamic font size based on the recognized name and face width.
    (Copied from refactored image_recognition.py)
    """
    font_size: int
    if name != "Unidentified" and name in known_names_for_font:
        try:
            name_idx = known_names_for_font.index(name)
            slope_val = slopes[name_idx]
            intercept_val = intercepts[name_idx]
            font_size = int(slope_val * face_width_for_font_calc + intercept_val)
        except (ValueError, IndexError):
            font_size = int(default_slope * face_width_for_font_calc + default_intercept)
    else:
        font_size = int(default_slope * face_width_for_font_calc + default_intercept)

    return max(10, min(font_size, 60)) # Clamp font size


def live_face_recognition(camera_index: int = 0) -> None:
    """
    Performs real-time face recognition using a webcam.

    Args:
        camera_index: Index of the camera to use (default is 0).
    """
    print("Initializing components for live recognition...")
    try:
        detector = FaceDetector(cascade_path=config.CASCADE_PATH)
        embedder = FaceEmbedder(model_path=config.MODEL_PATH)
        recognizer = FaceRecognizer(embeddings_path=config.EMBEDDINGS_PATH)

        if not os.path.exists(config.VARS_PATH):
            print(f"Error: Font parameters file not found at {config.VARS_PATH}.")
            print("Please run 'scripts/prepare_data.py' to generate it.")
            return
        font_params = np.load(config.VARS_PATH, allow_pickle=True)
        font_slopes = font_params['a']
        font_intercepts = font_params['b']
        known_font_names = list(font_params['c']) if 'c' in font_params else []
        if not known_font_names:
             print(f"Warning: No names found in font parameters file {config.VARS_PATH}. Using default font sizes.")

    except FileNotFoundError as e:
        print(f"Error initializing components: {e}.")
        print("One or more required files (cascade, model, embeddings, vars) might be missing.")
        print("Please ensure all paths in 'config.py' are correct and files exist.")
        print("You may need to run 'scripts/prepare_data.py' first.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        return

    print(f"Starting webcam (index {camera_index})...")
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}.")
        print("Please check if the camera is connected and the index is correct.")
        return

    print("Live recognition started. Press 'q' to quit.")
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame from camera. Exiting.")
            break

        # Original live_recognition used scaleFactor=1.2, minNeighbors=5
        detected_faces = detector.detect(frame, scale_factor=1.2, min_neighbors=5)

        annotated_frame = frame.copy()

        for (x_face, y_face, w_face, h_face) in detected_faces:
            dw = 0.1 * w_face  # Margin calculation from original
            # dh = 0.2 * h_face # Used by draw_face_rectangle internally

            face_crop = frame[y_face:y_face+h_face, x_face:x_face+w_face]
            if face_crop.size == 0:
                continue

            embedding = embedder.embed(face_crop)
            name, confidence = recognizer.recognize(embedding, threshold=1.0)
            # print(f"    Recognized: {name}, Conf: {confidence:.2f}") # Optional: for console logging

            # Calculate font size
            face_width_for_font_calc = int(((w_face + 2 * dw) / 3) * 2) # Original heuristic
            font_size = get_dynamic_font_size(
                name,
                face_width_for_font_calc,
                known_font_names,
                font_slopes,
                font_intercepts
            )

            annotated_frame = draw_face_rectangle(
                annotated_frame,
                (x_face, y_face, w_face, h_face),
                margin_dw=0.1,
                margin_dh=0.2
            )

            annotated_frame = draw_name_label(
                annotated_frame,
                name,
                (x_face, y_face, w_face, h_face),
                font_size=font_size,
                font_path=config.FONT_PATH,
                margin_dw=0.1,
                margin_dh=0.2
            )

        # Resize for display, original was (800,600)
        display_frame = cv2.resize(annotated_frame, (800, 600))
        cv2.imshow('Live Face Recognition (Press Q to quit)', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting live recognition...")
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")


if __name__ == '__main__':
    try:
        live_face_recognition()
    except FileNotFoundError as e: # Should be caught by initialization checks
        print(f"Initialization failed due to missing file: {e}")
        print("Ensure 'scripts/prepare_data.py' has been run and all paths in 'config.py' are correct.")
    except Exception as e:
        print(f"An unexpected error occurred in live recognition: {e}")
        import traceback
        traceback.print_exc()
