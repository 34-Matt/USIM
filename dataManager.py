import os
import regex as re
import cv2
import glob
import numpy as np
import pandas as pd

from typing import Tuple, List, Generator, Union

from dataManipulation.MymeekerStructure import MeekerObject

class SynergyDataManager():
    def __init__(self, img_size: Tuple[int], dataPath: str, structs: str, badJoints: list=[]):
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
        for bj in badJoints:
            self.jointDf.drop(bj, axis=1, inplace=True, errors="ignore")
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

    def randomSample(self, batchSize: int) -> Generator[None, None, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[int]]]:
        '''Generates a random sample of the data
        
        Args:
            batchSize [int]: The size of the batch to generate
            
        Yields:
            jointDf [DataFrame]: The joint data
            synergyDf [DataFrame]: The synergy data
            img [ndarray]: The image data
            indices [List[int]]: The indices of the data
        '''
        # Generate and shuffle all indices
        indice_list = np.arange(self.sample_count)
        np.random.shuffle(indice_list)
        while len(indice_list) > 0:
            if len(indice_list) > batchSize:
                indices = indice_list[:batchSize]
                indice_list = indice_list[batchSize:]
            else:
                indices = indice_list.copy()
                indice_list = []
        
            jointDf = self.jointDf.iloc[indices]
            synergyDf = self.synergyDf.iloc[indices]

            img = self.__image_sample(indices)

            yield jointDf, synergyDf, img, indices
    
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

        img = self.__image_sample(indices)

        return jointDf, synergyDf, img
    
    def __image_sample(self, indices: List[int]) -> np.ndarray:
        # Initialize image array
        img = np.zeros((len(indices), *self.img_size))

        # Iterate over each image and add to array
        for i, index in enumerate(indices):
            cur_img = cv2.imread(self.img_str.format(index), cv2.IMREAD_UNCHANGED)

            # Check image sizes
            if len(self.img_size) == len(cur_img.shape):
                img[i] = cv2.resize(cur_img, self.img_size)
            elif len(self.img_size) > len(cur_img.shape):
                for excess_shape in self.img_size[len(cur_img.shape):]:
                    if excess_shape != 1:
                        raise IndexError(f"Trying to print to an index of length {excess_shape} when image size does not contain dimension.")
                cur_img = cv2.resize(cur_img, self.img_size[:len(cur_img.shape)])
                img[i] = cur_img.reshape(self.img_size)

        return img
    
    def normalizeWrapper(self, joints:Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(joints, pd.DataFrame):
            return joints.apply(MeekerObject.normalize, axis=1)
        elif isinstance(joints, pd.Series):
            return MeekerObject.normalize(joints)
        else:
            raise TypeError("Expected the inputs to be a pandas object.")

    def denormalizeWrapper(self, joints:Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(joints, pd.DataFrame):
            return joints.apply(MeekerObject.denormalize, axis=1)
        elif isinstance(joints, pd.Series):
            return MeekerObject.denormalize(joints)
        else:
            raise TypeError("Expected the inputs to be a pandas object.")
