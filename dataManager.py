import os
import regex as re
import cv2
import glob
import numpy as np
import pandas as pd

from typing import Tuple, List, Generator

import imageTools
from dataManipulation.MymeekerStructure import MeekerObject

class SynergyDataManager():
    def __init__(self, img_size: Tuple[int], dataPath: str, structs: str):
        '''Initializes the DataManager

        Args:
            img_size [Tuple[int]]: The size of the image
            dataPath [str]: The path to the data. The folder should contain a csv and images.
            structs [str]: The structs to use
        '''
        self.img_size = img_size
        self.dataPath = dataPath
        self.structs = structs
        
        # Load Joint
        path_csv = glob.glob(os.path.join(dataPath, '*.csv'))
        if len(path_csv) != 1:
            raise FileNotFoundError(f"Expected 1 csv file in {dataPath}, but found {len(path_csv)}")
        self.jointDf = pd.read_csv(path_csv[0], index_col=0)
        
        badJoints = []
        if 'rh_WRJ1' in self.jointDf.columns:
            self.jointDf = self.jointDf.drop(columns=['rh_WRJ1'])
            badJoints.append('rh_WRJ1')
        if 'rh_WRJ2' in self.jointDf.columns:
            self.jointDf = self.jointDf.drop(columns=['rh_WRJ2']) 
            badJoints.append('rh_WRJ2')

        # Load meeker object
        self.meeker = MeekerObject(structs, badJoints)

        # Create Synergy
        self.synergyDf = self.jointDf.apply(self.meeker.forwardProject, axis=1)

        # Get image string
        tempimg = glob.glob(os.path.join(dataPath, '*.png'))[0]
        number_pattern = '([0-9]+)'
        numberIndex = None
        for numberIndex in re.finditer(number_pattern, tempimg):
            pass
        if numberIndex is None:
            raise ValueError(f"Could not find number in {tempimg}")
        number_start, number_end = numberIndex.start(0), numberIndex.end(0)
        
        img_start = tempimg[0:number_start]
        img_end = tempimg[number_end:]
        img_number_length = number_end - number_start
        self.img_str = img_start + '{:0' + str(img_number_length) + 'd}' + img_end

        img_count = len(glob.glob(os.path.join(dataPath, '*.png')))

        # Check that number of images match length of csv file
        if len(self.jointDf) != img_count:
            raise IndexError(f"Expected sizes to be the same, but got {self.jointDf.shape} for csv shape and {img_count} for number of images.")
        
        self.sample_count = img_count

    def randomSample(self, batchSize: int, noise: bool=False) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[int]], None, None]:
        '''Generates a random sample of the data
        
        Args:
            batchSize [int]: The size of the batch to generate
            noise [bool]: Whether to apply noise or not
            
        Yields:
            jointDf [DataFrame]: The joint data
            synergyDf [DataFrame]: The synergy data
            img [ndarray]: The image data
            indices [List[int]]: The indices of the data
        '''
        # Shuffle the indices
        indices = np.arange(self.sample_count)
        np.random.shuffle(indices)
        current_offset = 0

        # Iterate over samples of batchSizes
        while len(indices) - current_offset >= batchSize:
            cur_indices = indices[current_offset:current_offset+batchSize]
            current_offset += batchSize

            jointDf = self.jointDf.iloc[cur_indices]
            synergyDf = self.synergyDf.iloc[cur_indices]

            image = np.zeros((batchSize, *self.img_size))
            for i, index in enumerate(cur_indices):
                image[i,:,:,:] = self.__getImage(index, noise)

            yield jointDf, synergyDf, image, cur_indices
    
    def getSample(self, indices: List[int], noise: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        '''Gets a specific sample of the data
        matrix[matrix > saltprob] = 1
        Args:
            indices [List[int]]: The indices of the data
            noise [bool]: Whether to apply noise or not
            
        Returns:
            jointDf [DataFrame]: The joint data
            synergyDf [DataFrame]: The synergy data
            img [ndarray]: The image data
        '''
        jointDf = self.jointDf.iloc[indices]
        synergyDf = self.synergyDf.iloc[indices]

        img = np.zeros((len(indices), *self.img_size))
        for i, index in enumerate(indices):
            img[i,:,:,:] = self.__getImage(index, noise)

        return jointDf, synergyDf, img
        
    def __getImage(self, index:int, noise:bool) -> np.ndarray:
        # Load image
        img = cv2.imread(self.img_str.format(index), cv2.IMREAD_UNCHANGED)
        
        # Reshape image
        if len(img.shape) == len(self.img_size):
            img = cv2.resize(img,self.img_size)
        elif len(img.shape) == 2 and len(self.img_size) == 3:
            img = cv2.resize(img, self.img_size[:2])
            img = img.reshape(self.img_size)
        else:
            raise IndexError(f"Not sure what to do with image size of {img.shape} when mapping to size {self.img_size}")

        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        # Apply random noise to the data
        if noise:
            ## Random shifting
            x = 20 * np.random.sample() - 10
            y = 20 * np.random.sample() - 10
            theta = np.random.sample() * np.pi/36 # 5 degree

            transform = imageTools.get_translation(x, y, theta)
            img[:,:,0] = cv2.warpAffine(img[:,:,0], transform, img.shape[0:2])

            ## Gaussian
            noise = imageTools.create_gaussian(img.shape, 0.001)
            img += noise
            img = np.clip(img, 0, 1)

        return img
