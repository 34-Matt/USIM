import cv2
import numpy as np

from typing import Tuple

def get_translation(x: int, y: int, theta: float) -> np.ndarray:
    '''Creates the 2D translation matrix for image movement
    
    Args:
        x (int): The number of pixels to move the image along the x-axis
        y (int): The number of pixels to move the image along the y-axis
        theta (float): The angle to rotate the image counterclock-wise
    
    Returns:
        matrix (np.ndarray): The affline transform matrix
    '''

    matrix = np.zeros((2,3))

    # Translation
    matrix[0, 2] = x
    matrix[1, 2] = y

    # Rotation
    c, s = np.cos(theta), np.sin(theta)
    matrix[0:2, 0:2] = np.array(((c, -s), (s, c)))

    return matrix

def create_gaussian(shape:Tuple[int], var:float=0.5) -> np.ndarray:
    '''Generates a matrix with gaussian noise between the range of -1 and 1
    The gaussian noise has a mean of 0. The values are capped between -1 and 1.
    
    Args:
        shape (Tuple[int]): The matrix shape for the gaussian noise
        var (float): The variance for the gaussian distribution
    
    Returns:
        matrix (np.ndarray): The noise matrix
    '''
    matrix = np.random.normal(0, var, shape)
    return np.clip(matrix, -1, 1)

def create_saltpepper(shape:Tuple[int], saltprob: float, pepperprob: float) -> np.ndarray:
    '''Generates a matrix with salt [1] and pepper[-1] noise
    
    Args:
        shape (Tuple[int]): The matrix shape for the salt and pepper noise
        saltprob (float): The probability between 0 and 1 for a pixel to be salted
        pepperprob (float): The probability between 0 and 1 for a pixel to be peppered
    
    Returns:
        matrix (np.ndarray): The noise matrix
    '''
    matrix = np.random.uniform(-1, 1, shape)
    matrix[(matrix < saltprob) and (matrix > -pepperprob)] = 0
    matrix[matrix > saltprob] = 1
    matrix[matrix < -pepperprob] = -1
    return matrix

