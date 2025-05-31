import sys
import os
import unittest
import numpy as np

# Ensure the root directory is in sys.path for core.utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils import prewhiten, l2_normalize

class TestPrewhiten(unittest.TestCase):
    def test_3d_array(self):
        arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32) # 2x2x3
        whitened = prewhiten(arr)
        self.assertTrue(np.allclose(np.mean(whitened), 0.0, atol=1e-7), "Mean should be close to 0 for 3D")
        self.assertTrue(np.allclose(np.std(whitened), 1.0, atol=1e-7), "Std dev should be close to 1 for 3D")

    def test_4d_array(self):
        arr = np.random.rand(2, 10, 10, 3).astype(np.float32) * 255 # Batch of 2 images
        whitened = prewhiten(arr)
        for i in range(arr.shape[0]):
            self.assertTrue(np.allclose(np.mean(whitened[i]), 0.0, atol=1e-6), f"Mean of image {i} should be close to 0 (atol=1e-6)")
            # For prewhitening, std dev of each image (if whitened independently) might not be exactly 1
            # if the overall std_adj is based on the entire batch's characteristics or a global minimum.
            # The current prewhiten function calculates mean and std over axes (1,2,3) for 4D, so each image is whitened.
            self.assertTrue(np.allclose(np.std(whitened[i]), 1.0, atol=1e-6), f"Std dev of image {i} should be close to 1 (atol=1e-6)")


    def test_zero_std_dev(self):
        arr = np.ones((2, 2, 3), dtype=np.float32) * 5.0
        whitened = prewhiten(arr)
        # If mean is 5.0, and std_adj is max(0, 1/sqrt(size)), then (5-5)/std_adj = 0
        self.assertTrue(np.all(whitened == 0.0), "Output should be all zeros for zero std dev input")

    def test_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'Dimension should be 3 or 4.'):
            prewhiten(np.random.rand(5).astype(np.float32)) # 1D
        with self.assertRaisesRegex(ValueError, 'Dimension should be 3 or 4.'):
            prewhiten(np.random.rand(5, 5).astype(np.float32)) # 2D
        with self.assertRaisesRegex(ValueError, 'Dimension should be 3 or 4.'):
            prewhiten(np.random.rand(2,2,2,2,2).astype(np.float32)) # 5D

class TestL2Normalize(unittest.TestCase):
    def test_1d_vector(self):
        vec = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        normalized = l2_normalize(vec)
        self.assertTrue(np.allclose(np.linalg.norm(normalized), 1.0), "L2 norm of 1D vector should be 1")

    def test_2d_batch_vectors(self):
        batch = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        normalized = l2_normalize(batch)
        for i in range(normalized.shape[0]):
            self.assertTrue(np.allclose(np.linalg.norm(normalized[i]), 1.0), f"L2 norm of vector {i} in batch should be 1")

    def test_zero_vector(self):
        vec = np.zeros(5, dtype=np.float32)
        normalized = l2_normalize(vec)
        # np.linalg.norm(vec) is 0. Division by 0 results in nan for np.float32.
        # For np.float64, it might result in 0. Let's check for nan or zero.
        is_nan_or_zero = np.all(np.isnan(normalized)) or np.allclose(normalized, 0.0)
        self.assertTrue(is_nan_or_zero, "Normalized zero vector should be all NaNs or all zeros")


    def test_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'Input array must be 1D or 2D.'):
            l2_normalize(np.random.rand(2, 2, 2).astype(np.float32)) # 3D

if __name__ == '__main__':
    unittest.main()
