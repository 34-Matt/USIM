'''Meeker Structure
The functions in this file are used for processing processing data in the format described in
"Intuitive Hand Teleoperation by Novice Operators Using a Continuous Teleoperation Subspace" by C. Meeker.
The examples of the necessary information can be found in the structures folder.
'''

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from typing import List, Union


# Constants
INFO_COL = ['min', 'max', 'offset', 'flip']

# List of defined structures
__structFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'structures'))
humanHandStruct = os.path.join(__structFolder, 'rf_human_info.csv')
allegroHandStruct = os.path.join(__structFolder, 'rf_allegro_info.csv')
shadowHandStruct = os.path.join(__structFolder, 'rf_shadow_info.csv')
shadow4HandStruct = os.path.join(__structFolder, 'rf_shadow4_info.csv')
shadow3HandStruct = os.path.join(__structFolder, 'rf_shadow3_info.csv')

__dynamicStructFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'structures_dynamic'))
dynamicHumanHandStruct = os.path.join(__dynamicStructFolder, 'rf_human_info.csv')
dynamicAllegroHandStruct = os.path.join(__dynamicStructFolder, 'rf_allegro_info.csv')
dynamicShadowHandStruct = os.path.join(__dynamicStructFolder, 'rf_shadow_info.csv')
dynamicShadow4HandStruct = os.path.join(__dynamicStructFolder, 'rf_shadow4_info.csv')
dynamicShadow3HandStruct = os.path.join(__dynamicStructFolder, 'rf_shadow3_info.csv')

class MeekerObject:
    '''A class for storing the meeker structure of a hand'''
    
    def __init__(self, csvFile:str, badJoints:List[str] = [], useWeights:bool = False) -> None:
        '''Loads the meeker structure from a csv file
        
        Args:
            csvFile (str): a file location of the structure csvFile
            badJoints (List[str]): a list of joint names to remove from the structure
            useWeights (bool): whether to use the range weights when computing the synergy
        '''
        df = pd.read_csv(csvFile, index_col=0)
        for joint in badJoints:
            df = df.drop(joint)

        self.joint_col = list(df.index)
        df_columns = df.columns
        if not all ([c in df_columns for c in INFO_COL]):
            raise ValueError('Missing necessary columns in the structure file')
        self.synergy_col = list(df_columns.drop(INFO_COL))

        self.maxPos = df['max']
        self.minPos = df['min']
        
        self.A = df[self.synergy_col]
        if useWeights:
            self.prescale = self.computeWeights()
            self.A_scale = pd.Series(1, index=self.synergy_col)
        else:
            self.prescale = pd.Series(1, index=self.joint_col)
            self.A_scale = 1 / np.sum(self.A, axis=0)

        self.A_tensor = tf.convert_to_tensor(self.A, dtype=tf.float32)
        self.A_scale_tensor = tf.convert_to_tensor(self.A_scale, dtype=tf.float32)

    def forwardProject(self, curPos:pd.Series, isnormalized:bool) -> pd.Series:
        '''Projects a transform from joint space to teleoperation space
        
        Args:
            curPos (pd.Series): the joint angle for a given structure with shape [1xn]
            isnormalized (bool): whether the input is normalized (True) or not (False)
        
        Returns:
            synergy (pd.Series): the teleoperation position for a given structure with shape (1x3)
        '''
        if not self.checkValidInput(curPos):
            raise ValueError('Input is not valid for this structure')

        if isnormalized:
            normalizedPos = curPos
        else:
            normalizedPos = self.normalize(curPos)
        
        normalizedPos = normalizedPos * self.prescale
        synergy = normalizedPos.dot(self.A) * self.A_scale
        return synergy
    
    def forwardProjectTensorPred(self, curPos:tf.Tensor) -> tf.Tensor:
        '''Projects a transform from joint space to teleoperation space
        
        Args:
            curPos (tf.Tensor): the joint angle for a given structure with shape [1xn]
        
        Returns:
            synergy (tf.Tensor): the teleoperation position for a given structure with shape (1x3)
        '''
        synergy = tf.multiply(tf.matmul(curPos, self.A_tensor), self.A_scale_tensor)
        return synergy

    def invProject(self, synergy:pd.Series, normalizeout:bool) -> pd.Series:
        '''Projects a transform from teleoperation space to joint space
        
        Args:
            synergy (pd.Series): the teleoperation position for a given structure with shape (1x3)
            normalizeout (bool): Whether the output should be normalized joints (True) or not (False)
            
        Returns:
            curPos (pd.Series): the joint angle for a given structure with shape [1xn]
        '''
        if not self.checkValidSynergy(synergy, True):
            raise ValueError('Input is not valid for this structure', synergy)
        
        curPos = synergy.dot(self.A.T)

        if not normalizeout:
            curPos = self.denormalize(curPos)
        
        return curPos

    def normalize(self, curPos:pd.Series, clip:bool=True) -> pd.Series:
        '''Normalizes the joint angle to be between 0 and 1
        
        Args:
            curPos (pd.Series): the joint angle for a given structure with shape [1xn]
            clip (bool): whether to clip the normalized position to be between 0 and 1
        
        Returns:
            normalizedPos (pd.Series): the normalized joint angle for a given structure with shape [1xn]
        '''
        if not self.checkValidInput(curPos):
            raise ValueError('Input is not valid for this structure')
        
        normalizedPos = curPos.copy()
        normalizedPos = (normalizedPos - self.minPos) / (self.maxPos - self.minPos)
        if clip:
            normalizedPos = normalizedPos.clip(0, 1)
        return normalizedPos

    def denormalize(self, normalizedPos:pd.Series, clip:bool=True) -> pd.Series:
        '''Denormalizes the joint angle back to the original range
        
        Args:
            normalizedPos (pd.Series): the normalized joint angle for a given structure with shape [1xn]
            clip (bool): whether to clip the normalized position to be between 0 and 1
        
        Returns:
            curPos (pd.Series): the joint angle for a given structure with shape [1xn]
        '''
        if not self.checkValidInput(normalizedPos):
            raise ValueError('Input is not valid for this structure')
        
        curPos = normalizedPos.copy()
        if clip:
            curPos = curPos.clip(0, 1)
        
        curPos = curPos * (self.maxPos - self.minPos) + self.minPos
        return curPos
    
    def computeWeights(self) -> pd.Series:
        '''Compute weighted significance of each joint based on total range of motion of the corresponding synergy
        
        Returns:
            weights (pd.Series): the weighted significance of each joint with shape [1xn]
        '''
        motionrange = (self.maxPos - self.minPos).abs()

        synergyrange = motionrange.dot(self.A)
        synergymotion = self.invProject(synergyrange, True)
        
        weights = motionrange / synergymotion
        return weights
    
    def checkValidInput(self, curPos:pd.Series, exact=False) -> bool:
        '''Checks if labels for the input is valid for the given structure
        
        Args:
            curPos (pd.Series): the joint angle for a given structure with shape [1xn]
            exact (bool): whether to check if the input is exactly the same as the structure
        
        Returns:
            valid (bool): whether the input is valid
        '''
        valid = all([c in self.joint_col for c in curPos.index])
        if exact:
            return valid and len(curPos.index) == len(self.joint_col)
        else:
            return valid
        
    def series2array(self, series: pd.Series, isAngle: bool) -> np.array:
        '''Converts a series into a standardize array format

        Args:
            series (pd.Series): the series to convert
            isAngle (bool): whether the series is an angle or synergy
        
        Returns:
            array (np.ndarray): the standardized array format
        '''
        if isAngle:
            columns = self.joint_col
        else:
            columns = self.synergy_col
        
        array = np.zeros((1, len(columns)))
        for n, col in enumerate(columns):
            array[0, n] = series[col]
        return array

    def array2series(self, array: np.array, isAngle: bool) -> pd.Series:
        '''Converts a standardized array format into a series

        Args:
            array (np.ndarray): the standardized array format
            isAngle (bool): whether the series is an angle or synergy

        Returns:
            series (pd.Series): the series to convert
        '''
        if isAngle:
            columns = self.joint_col
        else:
            columns = self.synergy_col
        
        series = pd.Series(index=columns, dtype='float64')
        for n, col in enumerate(columns):
            series[col] = array[0,n]
        return series

    def matrix2df(self, matrix: np.array, isAngle: bool) -> pd.DataFrame:
        '''Converts a standardized array of array format into a dataframe
        '''
        if isAngle:
            columns = self.joint_col
        else:
            columns = self.synergy_col
        
        df = pd.DataFrame(data=matrix, columns=columns)
        return df
    
    def df2matrix(self, df: pd.DataFrame, isAngle: bool) -> np.array:
        '''Converts a dataframe into a standardized array of array format
        '''
        if isAngle:
            columns = self.joint_col
        else:
            columns = self.synergy_col
        
        matrix = np.zeros((df.shape[0], len(columns)))
        for n, col in enumerate(columns):
            matrix[:,n] = df[col].values
        return matrix
        
    def validateColumns(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Checks the dataframe and removes columns if they are not in this object.
        If the dataframe is missing columns, throws an error
        '''
        return df[self.joint_col]


    def checkValidSynergy(self, synergy:pd.Series, exact:bool=False) -> bool:
        '''Checks if labels for the input is valid for the given structure
        
        Args:
            synergy (pd.Series): the teleoperation position for a given structure with shape (1x3)
            exact (bool): whether to check if the input is exactly the same as the structure
        
        Returns:
            valid (bool): whether the input is valid
        '''
        valid = all([c in self.synergy_col for c in synergy.index])
        if exact:
            return valid and len(synergy.index) == len(self.synergy_col)
        else:
            return valid
