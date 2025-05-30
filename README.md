# Face Recognition Project

This project implements a face recognition system using a pre-trained FaceNet model (Keras implementation) and Haar cascades for face detection. It can perform recognition on static images and live webcam feeds.

## Project Structure

-   `config.py`: Contains all centralized file paths and configurations.
-   `core/`: Directory for core application logic.
    -   `utils.py`: Utility functions like image preprocessing (prewhiten, l2_normalize).
    -   `face_detector.py`: `FaceDetector` class for detecting faces in images.
    -   `face_embedder.py`: `FaceEmbedder` class for generating face embeddings using the FaceNet model.
    -   `face_recognizer.py`: `FaceRecognizer` class for comparing embeddings against a known database.
    -   `drawing_utils.py`: Utilities for drawing face rectangles and name labels on images.
-   `scripts/`: Directory for utility and data preparation scripts.
    -   `prepare_data.py`: Script to detect faces, generate embeddings, and calculate font parameters.
-   `image_recognition.py`: Script to perform face recognition on images in a directory.
-   `live_recognition.py`: Script to perform real-time face recognition using a webcam.
-   `data/`: Directory for all data.
    -   `cascade/haarcascade_frontalface_default.xml`: Haar cascade for face detection.
    -   `font/Calibri Regular.ttf`: Font file used for drawing names.
    -   `model/facenet_keras.h5`: The pre-trained FaceNet Keras model. **(Important: You need to provide this model file)**.
    -   `images/`: Place raw images here for face detection by `scripts/prepare_data.py`.
    -   `faces/`: Detected and cropped face images will be stored here by `scripts/prepare_data.py`.
    -   `arrays/`: Stores generated embeddings (`embeddings.npz`) and font parameters (`vars.npz`).
-   `test/`: Directory for test images. Processed images will be saved in `test/predicted/`.
-   `requirements.txt`: Lists all Python dependencies.

## Setup

1.  **Clone the repository.**
2.  **Obtain Model File**: Download or otherwise obtain the `facenet_keras.h5` model file and place it in the `data/model/` directory. This file is not included in the repository.
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    It's recommended to use a virtual environment.

## Usage

### 1. Prepare Data (Embeddings and Font Parameters)

This step is crucial and needs to be run first.

-   **Option A: Detect faces from raw images:**
    1.  Place your raw images (e.g., `PersonA.jpg`, `PersonB.png`) into the `data/images/` directory.
    2.  Run the `prepare_data.py` script:
        ```bash
        python scripts/prepare_data.py
        ```
    3.  The script will prompt you to choose an option. Select '2' to detect faces.
    4.  Detected faces will be saved in `data/faces/` (e.g., `PersonA_face_0.jpg`).
    5.  **Crucial**: Review the images in `data/faces/`. Delete any incorrect detections. Ensure that the filenames correctly represent the person's name (e.g., `PersonA_face_0.jpg`, `PersonA_face_1.jpg` for PersonA). The script uses the part of the filename before `_face_` as the person's name.
    6.  After cleanup, run `python scripts/prepare_data.py` again and choose option '1' ("Have faces already").
-   **Option B: Use existing cropped face images:**
    1.  Place your already cropped face images directly into the `data/faces/` directory. Ensure each filename is the person's name (e.g., `PersonA.jpg`, `PersonB.jpg`). If multiple images exist for the same person, ensure they are uniquely named but start with the person's actual name if you want them grouped, or handle naming according to how `prepare_data.py` extracts names (currently, it takes the filename before `.jpg` as the name if no `_face_` pattern is found). For best results with the current `prepare_data.py` when providing your own faces, name them like `Person Name.jpg`.
    2.  Run the `prepare_data.py` script:
        ```bash
        python scripts/prepare_data.py
        ```
    3.  Choose option '1' ("Have faces already").

This will generate `data/arrays/embeddings.npz` (containing face embeddings and names) and `data/arrays/vars.npz` (containing font calculation parameters).

### 2. Perform Recognition on Static Images

1.  Place images you want to test into the `test/` directory.
2.  Run the script:
    ```bash
    python image_recognition.py
    ```
3.  Annotated images will be saved in `test/predicted/`.

### 3. Perform Live Recognition

1.  Ensure your webcam is connected.
2.  Run the script:
    ```bash
    python live_recognition.py
    ```
3.  A window will appear showing the webcam feed with detected faces and names. Press 'q' to quit.

## Notes

-   The accuracy of the recognition depends heavily on the quality of the input images for embedding generation and the FaceNet model itself.
-   Ensure `config.py` points to the correct paths if you modify the directory structure further.
