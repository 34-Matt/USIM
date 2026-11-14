from scipy import ndimage
import numpy as np
import cv2

def randomRightShift(img: np.ndarray, maxShift: int, borderVal: float = 0) -> np.ndarray:
    '''Randomly shifts the center of the image to the right between the range of [-maxShift, maxShift]
    img: The image to shift
    maxShift: The maximum number to pixels to shift the image
    borderVal: The constant value to fill the border of the shifted image with
    '''
    shift = np.random.randint(-maxShift, maxShift+1)

    if shift > 0:
        img = img[:, shift:]
        img = cv2.copyMakeBorder(img,0,0,0,shift,cv2.BORDER_CONSTANT,value=borderVal)
    elif shift < 0:
        shift = abs(shift)
        img = img[:, :-shift]
        img = cv2.copyMakeBorder(img,0,0,shift,0,cv2.BORDER_CONSTANT,value=borderVal)
    
    return img

def randomUpShift(img: np.ndarray, maxShift: int, borderVal: float = 0) -> np.ndarray:
    '''Randomly shifts the center of the image to the top between the range of [-maxShift, maxShift]
    img: The image to shift
    maxShift: The maximum number to pixels to shift the image
    borderVal: The constant value to fill the border of the shifted image with
    '''
    shift = np.random.randint(-maxShift, maxShift+1)

    if shift > 0:
        img = img[:-shift, :]
        img = cv2.copyMakeBorder(img,shift,0,0,0,cv2.BORDER_CONSTANT,value=borderVal)
    elif shift < 0:
        shift = abs(shift)
        img = img[shift:, :]
        img = cv2.copyMakeBorder(img,0,shift,0,0,cv2.BORDER_CONSTANT,value=borderVal)
    
    return img

def randomRotate(img: np.ndarray, maxAngle: float, borderVal: float = 0) -> np.ndarray:
    '''Applies a random rotation to the image between the range [-maxAngle, maxAngle]
    img: The image to rotate
    maxAngle: The maximum value in degrees to rotate the image
    borderVal: The constant value to fill the border of the rotated image with
    '''
    shift = 2 * maxAngle * np.random.rand() - maxAngle
    return ndimage.rotate(img, shift, reshape=False, cval=borderVal)

def randomGaussianNoise(img: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    '''Applies random Gaussian noise to the image.
    img: The image to add noise to
    mean: The mean of the Gaussian noise
    sigma: The standard deviation of the Gaussian noise
    '''
    gauss = np.random.normal(mean,sigma,img.shape)
    return img + gauss

def addSpecklePattern(img: np.ndarray, rate: float, mag: float) -> np.ndarray:
    '''Applies random Gaussian noise to the image.
    img: The image to add noise to
    rate: The rate between 0 and 1 at which the speckle should occur
    mag: The magnitude of the speckle pattern
    '''
    speckle = np.random.random(img.shape) > rate
    return img + speckle
