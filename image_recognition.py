import sys
import os
import numpy as np
import cv2
from numpy.typing import NDArray
# No matplotlib.pyplot needed if using cv2.imwrite for saving

# Ensure the root directory is in sys.path for config and core imports
# This is not strictly necessary if image_recognition.py is in the root,
# but good practice if it might be moved or called from elsewhere.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config
# core.utils.l2_normalize is not directly needed here if recognizer handles it.
# from core.utils import l2_normalize
from core.face_detector import FaceDetector
from core.face_embedder import FaceEmbedder
from core.face_recognizer import FaceRecognizer
from core.drawing_utils import draw_face_rectangle, draw_name_label

# Global default values for font calculation if name is not found in vars
DEFAULT_SLOPE = 0.1974311
DEFAULT_INTERCEPT = 0.03397702412218706

def get_dynamic_font_size(
    name: str,
    face_width_for_font_calc: int, # Renamed for clarity from face_width_with_margin
    known_names_for_font: list[str],
    slopes: NDArray,
    intercepts: NDArray,
    default_slope: float = DEFAULT_SLOPE,
    default_intercept: float = DEFAULT_INTERCEPT
) -> int:
    """
    Calculates dynamic font size based on the recognized name and face width.

    Args:
        name: The recognized name.
        face_width_for_font_calc: The width heuristic value derived from face dimensions,
                                  e.g., ((w_face+2*dw)//3)*2 from original script.
        known_names_for_font: List of names for whom font parameters (slopes, intercepts) are known.
        slopes: Array of slopes corresponding to known_names_for_font.
        intercepts: Array of intercepts corresponding to known_names_for_font.
        default_slope: Slope to use if name is not found or "Unidentified".
        default_intercept: Intercept to use if name is not found or "Unidentified".

    Returns:
        The calculated font size (int).
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


def recognize_images_in_directory(input_dir: str, output_dir: str) -> None:
    """
    Recognizes faces in images within a directory and saves annotated images.

    Args:
        input_dir: Directory containing images to process.
        output_dir: Directory where processed images with annotations will be saved.
    """
    print("Initializing components...")
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

    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found.")
        return
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing images in '{input_dir}'.")
    # input("Press ENTER when you're ready to start processing...") # Removed for non-interactive run

    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')) and
                   not os.path.isdir(os.path.join(input_dir, f))]


    if not image_files:
        print(f"No image files found in '{input_dir}'.")
        return

    print(f"Found {len(image_files)} images to process.")

    for img_filename in image_files:
        img_path = os.path.join(input_dir, img_filename)
        print(f"\nProcessing image: {img_filename}...")

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  Could not read image: {img_filename}. Skipping.")
            continue

        detected_faces = detector.detect(frame, scale_factor=1.2, min_neighbors=10)

        if not detected_faces:
            print(f"  No faces detected in {img_filename}.")
        else:
            print(f"  Detected {len(detected_faces)} faces.")

        annotated_frame = frame.copy()

        for (x_face, y_face, w_face, h_face) in detected_faces:
            dw = 0.1 * w_face
            # dh = 0.2 * h_face # dh is used by draw_face_rectangle internally based on its margin_dh

            face_crop = frame[y_face:y_face+h_face, x_face:x_face+w_face]
            if face_crop.size == 0:
                print("  Skipping empty face crop.")
                continue

            embedding = embedder.embed(face_crop)
            name, confidence = recognizer.recognize(embedding, threshold=1.0)
            print(f"    Face recognized as: {name} (Distance: {confidence:.4f})")

            face_width_for_font_calc = int(((w_face + 2 * dw) / 3) * 2)

            font_size = get_dynamic_font_size(
                name,
                face_width_for_font_calc,
                known_font_names,
                font_slopes,
                font_intercepts
            )
            # print(f"    Calculated font size: {font_size} for name '{name}'") # Optional: for debugging

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

        output_image_path = os.path.join(output_dir, img_filename)
        try:
            cv2.imwrite(output_image_path, annotated_frame)
            print(f"  Saved annotated image to: {output_image_path}")
        except Exception as e:
            print(f"  Error saving image {output_image_path}: {e}")

    print("\nProcessing complete.")
    print(f"Annotated images saved in '{output_dir}'.")


if __name__ == '__main__':
    try:
        # Ensure test directories exist for the example run
        os.makedirs(config.TEST_IMAGES_DIR, exist_ok=True)
        os.makedirs(config.PREDICTED_IMAGES_DIR, exist_ok=True)

        # Example: Add a dummy image to test if TEST_IMAGES_DIR is empty
        # This part should ideally be handled by user or setup script
        dummy_image_path = os.path.join(config.TEST_IMAGES_DIR, "dummy_image.png")
        if not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(config.TEST_IMAGES_DIR)):
            # Create a simple black image if no test images are present
            if not os.path.exists(dummy_image_path):
                dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(dummy_image_path, dummy_img)
                print(f"Created a dummy image at {dummy_image_path} for testing.")

        recognize_images_in_directory(config.TEST_IMAGES_DIR, config.PREDICTED_IMAGES_DIR)

    except FileNotFoundError as e:
        print(f"Critical file/directory not found during setup or run: {e}")
        print("Please ensure paths in 'config.py' are correct and all necessary data files exist.")
        print("You might need to run 'scripts/prepare_data.py' to generate embeddings and font parameters.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
