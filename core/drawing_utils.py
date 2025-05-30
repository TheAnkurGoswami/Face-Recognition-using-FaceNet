import sys
import os
# Ensure the root directory is in sys.path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.typing import NDArray

import config # For FONT_PATH and VARS_PATH

def draw_face_rectangle(
    image: NDArray,
    face_coords: tuple[int, int, int, int],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    margin_dw: float = 0.1,
    margin_dh: float = 0.2
) -> NDArray:
    """
    Draws a rectangle around a detected face with specified margins.

    Args:
        image: The image (NumPy array, BGR) on which to draw.
        face_coords: A tuple (x, y, w, h) representing the detected face box.
        color: Color of the rectangle (B, G, R). Defaults to green.
        thickness: Thickness of the rectangle lines.
        margin_dw: Proportional width margin to add to the face box (e.g., 0.1 for 10%).
        margin_dh: Proportional height margin to add to the face box (e.g., 0.2 for 20%).

    Returns:
        The image (NumPy array) with the rectangle drawn.
    """
    x, y, w, h = face_coords
    dw = int(margin_dw * w) # Calculate pixel margin for width
    dh = int(margin_dh * h) # Calculate pixel margin for height

    pt1 = (x - dw, y - dh)
    pt2 = (x + w + dw, y + h + dh)

    # cv2.rectangle handles clipping if points are outside image boundaries.
    cv2.rectangle(image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
    return image

def draw_name_label(
    image: NDArray,
    name: str,
    face_coords: tuple[int, int, int, int],
    font_size: int, # Added font_size parameter
    font_path: str = config.FONT_PATH,
    font_color: tuple[int, int, int] = (255, 0, 0),
    bg_color: tuple[int, int, int] = (0, 255, 0),
    margin_dw: float = 0.1,
    margin_dh: float = 0.2
) -> NDArray:
    """
    Draws a name label above the face rectangle using the specified font size.

    Args:
        image: The image (NumPy array, BGR) on which to draw.
        name: The name text to display.
        face_coords: A tuple (x, y, w, h) of the original detected face (before margins).
        font_size: The font size to use for the label.
        font_path: Path to the TrueType font file.
        font_color: Color of the text (B, G, R for consistency with cv2, converted to RGB for PIL).
        bg_color: Color of the background rectangle for the text (B, G, R, converted for PIL).
        margin_dw: Proportional width margin (consistent with face rectangle).
        margin_dh: Proportional height margin (consistent with face rectangle).

    Returns:
        The image (NumPy array) with the name label drawn.
    """
    x, y, w, h = face_coords
    dw = int(margin_dw * w)
    # dh = int(margin_dh * h) # Not directly used for label y-pos, which is relative to face_coords

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # print(f"Warning: Font file not found at {font_path} or cannot be opened. Using default PIL font.")
        font = ImageFont.load_default()

    # Convert cv2 image (BGR) to PIL image (RGB) for text drawing
    if image.dtype != np.uint8: # Ensure image is uint8 for PIL
        if np.max(image) <= 1.0: # Potentially float image in range [0,1]
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Using PIL's getsize method for older Pillow, or textbbox/getlength for newer.
    # getsize is simpler for this context if available and sufficient.
    try:
        text_width, text_height = draw.textsize(name, font=font) # Deprecated in Pillow 10
    except AttributeError: # Fallback for Pillow 10+
        bbox = draw.textbbox((0,0), name, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

    text_size = (text_width, text_height)


    # Position of the text label background, above the original face rectangle (y - dh_rect)
    # The main rectangle's top-left y is y_rect = y - dh_main_rect
    # Text label top-left y should be y_rect - text_height = y - dh_main_rect - text_height
    y_rect_top = y - int(margin_dh * h) # Top y of the green face rectangle

    label_bg_pt1 = (x - dw, y_rect_top - text_size[1] - 5) # 5px padding above text
    label_bg_pt2 = (x - dw + text_size[0] + 10, y_rect_top -5) # 10px padding for width, 5px for bottom of text

    # PIL uses RGB, so colors from config (assumed BGR) need conversion
    pil_bg_color = (bg_color[2], bg_color[1], bg_color[0]) # BGR to RGB
    pil_font_color = (font_color[2], font_color[1], font_color[0]) # BGR to RGB

    draw.rectangle([label_bg_pt1, label_bg_pt2], fill=pil_bg_color)

    text_pt = (x - dw + 5, y_rect_top - text_size[1] - 5) # 5px padding for text start
    draw.text(text_pt, name, font=font, fill=pil_font_color)

    drawn_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return drawn_image

if __name__ == '__main__':
    print("Running drawing_utils example...")
    try:
        # This import is fine as face_detector also handles its own sys.path for config
        from core.face_detector import FaceDetector

        # Check if necessary files exist
        if not os.path.exists(config.CASCADE_PATH):
            print(f"Cascade file not found: {config.CASCADE_PATH}. Skipping example.")
        # FONT_PATH check is good, but PIL has a default fallback. vars.npz is optional for this example.
        elif not os.path.exists(config.FONT_PATH):
             print(f"Warning: Font file not found at {config.FONT_PATH}. Text might use default font.")

        # Proceed even if font is missing, PIL will use a fallback.

        print(f"Loading FaceDetector with cascade from: {config.CASCADE_PATH}")
        detector = FaceDetector()

        test_image_path = './data/images/Chris Hemsworth.jpg'
        if not os.path.exists(test_image_path):
            print(f"Test image not found: {test_image_path}. Skipping drawing part of example.")
        else:
            print(f"Loading sample image: {test_image_path}")
            sample_image = cv2.imread(test_image_path)

            if sample_image is not None:
                print("Detecting faces...")
                detected_faces = detector.detect(sample_image, min_neighbors=5)

                if detected_faces:
                    print(f"Detected {len(detected_faces)} faces.")

                    face_coords_original = detected_faces[0] # (x,y,w,h)

                    print(f"Drawing rectangle and label for face at: {face_coords_original}")

                    img_copy = sample_image.copy()
                    img_with_rect = draw_face_rectangle(img_copy, face_coords_original)

                    # Using a placeholder name and a fixed font_size for the example.
                    example_font_size = 20
                    img_with_label = draw_name_label(img_with_rect, "Chris Hemsworth", face_coords_original, font_size=example_font_size)

                    print("Displaying image with drawings... Press any key to close.")
                    # Create a directory for test output if it doesn't exist
                    os.makedirs("./test/predicted", exist_ok=True)
                    output_path = "./test/predicted/drawing_utils_example.jpg"
                    cv2.imwrite(output_path, img_with_label)
                    print(f"Saved example image to {output_path}")

                    # cv2.imshow("Face Detection with Label", img_with_label)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    print("Example finished. Check the saved image.")

                else:
                    print("No faces detected in the sample image.")
            else:
                print(f"Failed to load sample image from {test_image_path}.")

    except FileNotFoundError as fnf_error:
        print(f"File not found error in example (should have been caught by specific checks): {fnf_error}")
    except ImportError as ie:
        print(f"Import error: {ie}. Make sure all core modules and dependencies (like PIL) are accessible.")
    except Exception as e:
        print(f"An error occurred during drawing_utils example usage: {e}")
        import traceback
        traceback.print_exc()
