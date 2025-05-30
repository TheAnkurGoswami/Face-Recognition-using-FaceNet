import numpy as np
from numpy.typing import NDArray

def prewhiten(x: NDArray) -> NDArray:
    """Prewhitens an image by subtracting the mean and dividing by the standard deviation.

    Args:
        x: Input image array. Can be 3D (single image) or 4D (batch of images).

    Returns:
        Prewhitened image array.

    Raises:
        ValueError: If the input array dimension is not 3 or 4.
    """
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4.')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x: NDArray) -> NDArray:
    """Normalizes a vector or a batch of vectors to unit length using L2 norm.

    Args:
        x: Input array. Can be 1D (single vector) or 2D (batch of vectors).

    Returns:
        L2 normalized array.

    Raises:
        ValueError: If the input array dimension is not 1 or 2.
    """
    if x.ndim == 2: # Batch of embeddings
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / norm
    elif x.ndim == 1: # Single embedding
        norm = np.linalg.norm(x)
        return x / norm
    else:
        raise ValueError('Input array must be 1D or 2D.')
