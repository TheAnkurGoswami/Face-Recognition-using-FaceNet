import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2 # Only needed for cv2.error if we want to mock that specifically, and for COLOR_BGR2GRAY

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.face_detector import FaceDetector
import config # To access config.CASCADE_PATH for a valid path structure, though not strictly used by mocks

class TestFaceDetector(unittest.TestCase):

    @patch('cv2.CascadeClassifier')
    def test_init_success(self, MockCascadeClassifier):
        # Arrange: Mock CascadeClassifier to simulate successful loading
        mock_cascade_instance = MockCascadeClassifier.return_value
        mock_cascade_instance.empty.return_value = False # Simulate not empty

        # Act: Create FaceDetector instance
        try:
            detector = FaceDetector(cascade_path='dummy_valid_path.xml')
            # Assert: CascadeClassifier was called with the path and no exception raised
            MockCascadeClassifier.assert_called_once_with('dummy_valid_path.xml')
            self.assertIsNotNone(detector.face_cascade)
        except Exception as e:
            self.fail(f"Initialization failed unexpectedly: {e}")


    @patch('cv2.CascadeClassifier')
    def test_init_failure_cascade_empty(self, MockCascadeClassifier):
        # Arrange: Mock CascadeClassifier to simulate it being empty after loading
        mock_cascade_instance = MockCascadeClassifier.return_value
        mock_cascade_instance.empty.return_value = True # Simulate empty

        # Act & Assert: Expect an IOError
        with self.assertRaisesRegex(IOError, "Failed to load Haar cascade from dummy_empty_path.xml"):
            FaceDetector(cascade_path='dummy_empty_path.xml')
        MockCascadeClassifier.assert_called_once_with('dummy_empty_path.xml')

    # Using @patch.object to mock methods of an instance after it's created,
    # or to mock __init__ itself to prevent it from running for certain tests.
    # For detect method tests, we want to bypass the actual __init__ that loads files.
    @patch.object(FaceDetector, '__init__', lambda self, cascade_path=None: None) # Bypass __init__
    def test_detect_faces_found(self):
        # Arrange
        detector = FaceDetector() # Init is bypassed, cascade_path is irrelevant here

        mock_cv2_cascade = MagicMock()
        # Define the return value for detectMultiScale: list of (x, y, w, h) tuples as numpy array
        mock_cv2_cascade.detectMultiScale.return_value = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        detector.face_cascade = mock_cv2_cascade # Assign the mock object

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        result = detector.detect(dummy_image)

        # Assert
        mock_cv2_cascade.detectMultiScale.assert_called_once()
        args, kwargs = mock_cv2_cascade.detectMultiScale.call_args
        # args[0] should be the grayscaled image. We can check its shape or type if needed.
        self.assertEqual(kwargs.get('scaleFactor'), 1.1) # Default scaleFactor
        self.assertEqual(kwargs.get('minNeighbors'), 5) # Default minNeighbors
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (10, 20, 30, 40)) # Ensure conversion to int tuple
        self.assertEqual(result[1], (50, 60, 70, 80))

    @patch.object(FaceDetector, '__init__', lambda self, cascade_path=None: None) # Bypass __init__
    def test_detect_no_faces_found(self):
        # Arrange
        detector = FaceDetector()
        mock_cv2_cascade = MagicMock()
        mock_cv2_cascade.detectMultiScale.return_value = np.array([]) # Empty array for no faces
        detector.face_cascade = mock_cv2_cascade

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        result = detector.detect(dummy_image)

        # Assert
        self.assertEqual(len(result), 0)

    @patch.object(FaceDetector, '__init__', lambda self, cascade_path=None: None) # Bypass __init__
    def test_detect_empty_image(self):
        # Arrange
        detector = FaceDetector()
        # No need to mock detectMultiScale as it shouldn't be called for None image

        # Act
        result_none = detector.detect(None)

        # Assert
        self.assertEqual(result_none, [])

        # Test with an empty numpy array if the function is expected to handle it
        # The current implementation does not explicitly check for image.size == 0 after None check,
        # cv2.cvtColor would error. This test focuses on the None case.
        # If specific handling for empty np.array is desired, the main code and this test would adapt.


    @patch.object(FaceDetector, '__init__', lambda self, cascade_path=None: None) # Bypass __init__
    @patch('cv2.cvtColor') # Mock cvtColor
    def test_detect_grayscale_conversion_called(self, mock_cvt_color):
        # Arrange
        detector = FaceDetector()
        mock_cv2_cascade = MagicMock()
        mock_cv2_cascade.detectMultiScale.return_value = np.array([])
        # Provide a dummy grayscale image as the return for cvtColor
        mock_cvt_color.return_value = np.zeros((100,100), dtype=np.uint8)
        detector.face_cascade = mock_cv2_cascade

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        detector.detect(dummy_image)

        # Assert
        mock_cvt_color.assert_called_once_with(dummy_image, cv2.COLOR_BGR2GRAY)
        # Check that detectMultiScale was called with the (mocked) grayscaled image
        args, _ = mock_cv2_cascade.detectMultiScale.call_args
        self.assertIs(args[0], mock_cvt_color.return_value)


if __name__ == '__main__':
    unittest.main()
