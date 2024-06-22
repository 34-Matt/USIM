import os
import re
import cv2
import glob
import numpy as np
import pandas as pd

from typing import Tuple, List

from dataManipulation.MymeekerStructure import MeekerObject

class SynergyDataManager():
    def __init__(self, img_size: Tuple[int], dataPath: str, structs: str, badJoints: list=None):
        '''Initializes the DataManager

        Args:
            img_size [Tuple[int]]: The size of the image
            dataPath [str]: The path to the data. The folder should contain a csv and images.
            structs [str]: The structs to use
        '''
        self.img_size = img_size
        self.dataPath = dataPath
        self.structs = structs

        # Load meeker object
        self.meeker = MeekerObject(structs, badJoints)

        # Read csv
        path_csv = glob.glob(os.path.join(dataPath, '*.csv'))
        if len(path_csv) != 1:
            raise FileNotFoundError(f"Expected 1 csv file in {dataPath}, but found {len(path_csv)}")
        
        self.jointDf = pd.read_csv(path_csv[0], index_col=0)
        self.synergyDf = self.jointDf.apply(self.meeker.forwardProject, axis=1)

        # Get image string
        tempimg = glob.glob(os.path.join(dataPath, '*.png'))[0]
        number_pattern = '([0-9]+)'
        for numberIndex in re.finditer(number_pattern, tempimg):
            pass
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

    def randomSample(self, batchSize: int) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[int]]:
        '''Generates a random sample of the data
        
        Args:
            batchSize [int]: The size of the batch to generate
            
        Returns:
            jointDf [DataFrame]: The joint data
            synergyDf [DataFrame]: The synergy data
            img [ndarray]: The image data
            indices [List[int]]: The indices of the data
        '''
        indices = np.random.choice(self.sample_count, batchSize)
        jointDf = self.jointDf.iloc[indices]
        synergyDf = self.synergyDf.iloc[indices]

        img = np.zeros((batchSize, *self.img_size))
        for i, index in enumerate(indices):
            img[i] = cv2.imread(self.img_str.format(index), cv2.IMREAD_UNCHANGED)

        return jointDf, synergyDf, img, indices
    
    def getSample(self, indices: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        '''Gets a specific sample of the data
        
        Args:
            indices [List[int]]: The indices of the data
            
        Returns:
            jointDf [DataFrame]: The joint data
            synergyDf [DataFrame]: The synergy data
            img [ndarray]: The image data
        '''
        jointDf = self.jointDf.iloc[indices]
        synergyDf = self.synergyDf.iloc[indices]

        img = np.zeros((len(indices), *self.img_size))
        for i, index in enumerate(indices):
            img[i] = cv2.imread(self.img_str.format(index), cv2.IMREAD_UNCHANGED)

        return jointDf, synergyDf, img
        
