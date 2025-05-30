import sys
import os
# Ensure the root directory is in sys.path for config and core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2 # For image loading in face detection, and resizing in general
from PIL import Image, ImageFont # For image loading and font calculations
from sklearn.linear_model import LinearRegression
from numpy.typing import NDArray

import config
from core.utils import prewhiten, l2_normalize
from core.face_detector import FaceDetector
from core.face_embedder import FaceEmbedder


def detect_and_save_faces(images_dir: str, faces_output_dir: str, detector: FaceDetector) -> None:
    """
    Detects faces in images from a source directory and saves cropped faces to an output directory.

    Args:
        images_dir: Path to the directory containing source images.
        faces_output_dir: Path to the directory where detected face crops will be saved.
        detector: An instance of FaceDetector.
    """
    if not os.path.exists(images_dir):
        print(f"Source images directory not found: {images_dir}")
        return

    os.makedirs(faces_output_dir, exist_ok=True)
    print(f"Looking for images in: {images_dir}")
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image_name_without_ext = os.path.splitext(image_file)[0]

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Could not read image: {image_path}")
            continue

        print(f"Detecting faces in {image_file}...")
        # Using min_neighbors=10 as a potentially stricter detection from original image_recognition.py
        detected_face_coords = detector.detect(img_bgr, min_neighbors=10)

        if not detected_face_coords:
            print(f"  No faces detected in {image_file}.")
            continue

        print(f"  Found {len(detected_face_coords)} faces in {image_file}.")
        for i, (x, y, w, h) in enumerate(detected_face_coords):
            face_crop = img_bgr[y:y+h, x:x+w]
            # Resize face crop to 160x160 before saving, or ensure it's done before embedding
            # The original generate_data loads and resizes. Let's save as is and resize on load.

            # Improved face naming: OriginalName_face_0.jpg
            saved_face_filename = f"{image_name_without_ext}_face_{i}.jpg"
            saved_face_path = os.path.join(faces_output_dir, saved_face_filename)

            try:
                cv2.imwrite(saved_face_path, face_crop)
                print(f"    Saved face {i+1} to {saved_face_path}")
            except Exception as e:
                print(f"    Error saving face {i+1} from {image_file}: {e}")
    print(f"Face detection and saving complete. Please check the '{faces_output_dir}' directory.")


def calculate_font_parameters(names: list[str], font_path: str) -> tuple[NDArray, NDArray]:
    """
    Calculates font slope and intercept for a list of names using LinearRegression.

    Args:
        names: A list of unique names (strings).
        font_path: Path to the TrueType font file.

    Returns:
        A tuple containing two NumPy arrays: (slopes, intercepts).
    """
    slopes = []
    intercepts = []

    if not os.path.exists(font_path):
        print(f"Warning: Font file not found at {font_path}. Cannot calculate font parameters.")
        # Return empty arrays or arrays of a default value if names list is not empty
        if names:
            return np.array([0.0] * len(names)), np.array([20.0] * len(names)) # Default slope 0, intercept 20 (font size)
        else:
            return np.array([]), np.array([])

    print("Calculating font parameters for text scaling...")
    for name_text in names:
        x_data = [] # font sizes
        y_data = [] # text widths corresponding to font sizes
        for font_size_pts in range(1, 100): # Iterate through font sizes 1 to 99
            try:
                font = ImageFont.truetype(font_path, font_size_pts)
                # Use textbbox for modern Pillow, getsize for older or as fallback
                try:
                    text_width = font.getlength(name_text)
                except AttributeError: # Pillow < 8.0.0 or some specific versions might need getsize
                    text_width = font.getsize(name_text)[0]

                x_data.append(font_size_pts)
                y_data.append(text_width)
            except Exception as e:
                print(f"  Warning: Could not get size for font {font_path} at size {font_size_pts} for name '{name_text}'. Error: {e}")
                continue

        if not x_data or not y_data or len(x_data) < 2 : # Need at least 2 points for linear regression
            print(f"  Warning: Not enough data points to calculate font parameters for '{name_text}'. Using defaults.")
            slopes.append(0.0) # Default slope
            intercepts.append(20.0) # Default intercept (e.g. fixed font size 20)
            continue

        # Reshape for sklearn: y_data (text widths) are features, x_data (font sizes) is target
        lin_reg = LinearRegression().fit(np.array(y_data).reshape(-1, 1), np.array(x_data))
        slopes.append(lin_reg.coef_[0])
        intercepts.append(lin_reg.intercept_)
        print(f"  Calculated for '{name_text}': slope={lin_reg.coef_[0]:.4f}, intercept={lin_reg.intercept_:.4f}")

    return np.array(slopes), np.array(intercepts)


def generate_embeddings_and_font_data() -> None:
    """
    Main function to orchestrate face detection (optional), embedding generation,
    and font parameter calculation. Saves embeddings, names, and font parameters to .npz files.
    """
    print("Starting data preparation process...")

    # Ensure data directories exist
    os.makedirs(config.FACES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.EMBEDDINGS_PATH), exist_ok=True) # For data/arrays/
    os.makedirs(os.path.dirname(config.VARS_PATH), exist_ok=True)       # For data/arrays/

    # Hardcode user_choice to '1' to bypass input() for non-interactive environment
    user_choice = '1'
    print(f"Simulating user choice: {user_choice}") # Optional: for logging

    if user_choice == '2':
        print("\n--- Face Detection Phase ---")
        if not os.path.exists(config.CASCADE_PATH):
            print(f"Error: Haar cascade file not found at {config.CASCADE_PATH}. Cannot proceed with face detection.")
            return
        try:
            detector = FaceDetector(cascade_path=config.CASCADE_PATH)
            detect_and_save_faces(config.IMAGES_DIR, config.FACES_DIR, detector)
            print("\nIMPORTANT: Please review the images in 'data/faces/'.")
            print("Delete any incorrect detections or unwanted faces.")
            print("Ensure filenames correctly represent the person's name (e.g., 'PersonName_face_0.jpg').")
            # input("Press ENTER to continue after you have reviewed and cleaned the 'data/faces' directory.\n") # Bypass this input too
            print("Simulating pressing ENTER after face review.")
        except Exception as e:
            print(f"Error during face detection phase: {e}")
            return

    elif user_choice != '1':
        print("Invalid choice. Exiting.")
        return

    print("\n--- Embedding Generation Phase ---")
    face_files = os.listdir(config.FACES_DIR)
    face_files = sorted([f for f in face_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not face_files:
        print(f"No face images found in {config.FACES_DIR}. Cannot generate embeddings.")
        print("If you ran face detection, ensure it completed successfully and you reviewed the files.")
        print("If you chose option 1, ensure your face images are in the correct directory.")
        return

    pil_faces = []
    names_from_filenames = []
    print(f"Loading faces from {config.FACES_DIR}...")
    for f_name in face_files:
        # Extract name: "PersonName_face_0.jpg" -> "PersonName"
        base_name = f_name.rsplit('_face_', 1)[0]
        names_from_filenames.append(base_name)

        image_path = os.path.join(config.FACES_DIR, f_name)
        try:
            img = Image.open(image_path).resize((160, 160))
            # Convert to BGR numpy array for prewhitening and embedding if model expects BGR
            # FaceNet typically trained on RGB, but original fx.prewhiten worked on arrays.
            # Let's assume prewhiten handles RGB format correctly.
            img_np = np.array(img)
            if img_np.ndim == 2: # Grayscale
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB) # Convert to RGB
            elif img_np.shape[2] == 4: # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB) # Convert to RGB

            pil_faces.append(img_np)
            print(f"  Loaded and resized {f_name} (Name: {base_name})")
        except Exception as e:
            print(f"  Error loading or processing {f_name}: {e}")
            continue

    if not pil_faces:
        print("No faces were successfully loaded. Exiting.")
        return

    # Convert list of PIL images (as numpy arrays) to a single numpy array for processing
    faces_np_array = np.array(pil_faces)

    print("Prewhitening faces...")
    prewhitened_faces = prewhiten(faces_np_array)

    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: FaceNet model not found at {config.MODEL_PATH}. Cannot generate embeddings.")
        return

    print(f"Loading FaceNet model from {config.MODEL_PATH}...")
    try:
        # Embedder class loads model, but for batch prediction, direct access might be intended by original.
        # Using embedder.model.predict for consistency if embedder handles model loading well.
        embedder = FaceEmbedder(model_path=config.MODEL_PATH)
        print("Generating embeddings...")
        # The model.predict expects batch of shape (None, 160, 160, 3)
        raw_embeddings = embedder.model.predict(prewhitened_faces)
    except Exception as e:
        print(f"Error during model loading or embedding generation: {e}")
        return

    print("L2 normalizing embeddings...")
    normalized_embeddings = l2_normalize(raw_embeddings)

    unique_names_list = sorted(list(set(names_from_filenames)))
    names_array_for_saving = np.array(names_from_filenames) # This saves name for each embedding

    print("\n--- Font Parameter Calculation Phase ---")
    # Calculate font parameters for the unique names that will be used for display
    slopes, intercepts = calculate_font_parameters(unique_names_list, config.FONT_PATH)

    # Saving data
    print("\n--- Saving Data ---")
    try:
        np.savez_compressed(config.EMBEDDINGS_PATH, a=normalized_embeddings, b=names_array_for_saving)
        print(f"Embeddings and corresponding names saved to: {config.EMBEDDINGS_PATH}")

        # VARS_PATH should store slopes and intercepts for the *unique_names_list*
        # This implies that when recognizing, the name is found, then its index in unique_names_list
        # is used to get the slope/intercept. The original FaceRecognizer might need adjustment
        # or this part needs careful alignment.
        # For now, assume recognition will map recognized name to an index in unique_names_list.
        np.savez_compressed(config.VARS_PATH, a=slopes, b=intercepts, c=np.array(unique_names_list)) # Save unique names with their slopes/intercepts
        print(f"Font parameters (slopes, intercepts, and unique names) saved to: {config.VARS_PATH}")

        print("\nData preparation complete!")
    except Exception as e:
        print(f"Error saving data: {e}")


if __name__ == '__main__':
    try:
        generate_embeddings_and_font_data()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
