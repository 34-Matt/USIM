'''Meeker Structure
The functions in this file are used for processing processing data in the format described in
"Intuitive Hand Teleoperation by Novice Operators Using a Continuous Teleoperation Subspace" by C. Meeker.
The examples of the necessary information can be found in the structures folder.
'''

import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import minimize

# List of defined structures
__structFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'structures'))
humanStruct = os.path.join(__structFolder, 'rf_human_info.csv')
shadowHandStruct = os.path.join(__structFolder, 'rf_shadow_info.csv')
allegroHandStruct = os.path.join(__structFolder, 'rf_allegro_info.csv')

class MeekerObject:

    df = None
    forwardScale = None
    invScale = None

    
    def __init__(self, csvFile, badJoints = []):
        '''Loads a csv file containing the baseline information
        csvFile: a file location of the structure csvFile
        badJoints: a list of joint names to remove
        
        returns dataframe with the completed information, forward scaling factor, and inverse scaling factor
        '''
        self.df = pd.read_csv(csvFile, index_col=0)
        for joint in badJoints:
            self.df = self.df.drop(joint)
        
        # Normalize Projection
        alpha = np.sqrt(np.sum(self.df['alpha']))
        sigma = np.sqrt(np.sum(self.df['sigma']))
        epsilon = np.sqrt(np.sum(self.df['epsilon']))
        self.df['alpha_norm'] = self.df['alpha'] / alpha
        self.df['sigma_norm'] = self.df['sigma'] / sigma
        self.df['epsilon_norm'] = self.df['epsilon'] / epsilon

        # Scaling Factor
        maxPos = self.df['max']
        minPos = self.df['min']
        self.forwardScale = np.array([[1, 1, 1]])

        maxSynergy = self.forwardProject(maxPos)
        minSynergy = self.forwardProject(minPos)

        self.forwardScale = np.array([[_scaling_apply(maxSynergy[0,0], minSynergy[0,0]), _scaling_apply(maxSynergy[0,1], minSynergy[0,1]), _scaling_apply(maxSynergy[0,2], minSynergy[0,2])]])
        self.invScale = np.array([[1/self.forwardScale[0,0], 1/self.forwardScale[0,1], 1/self.forwardScale[0,2]]])

        # Get min and max Synergy
        self.maxSynergy = self.forwardProject(maxPos)
        self.minSynergy = self.forwardProject(minPos)

    def forwardProject(self, curPos, normalize_synergy=False):
        '''Projects a transform from joint space to teleoperation space
        curPos: the joint angle for a given structure with shape [1xn]
        normalize_synergy: boolean to normalize synergy (True) or not (False)
        '''
        A = self.df[['alpha_norm', 'sigma_norm', 'epsilon_norm']].values
        offset = self.df['offset'].values
        curPos = self.normalize(curPos)

        synergy = np.multiply(np.dot(curPos - offset, A), self.forwardScale)

        if normalize_synergy:
            return np.clip(np.divide((synergy - self.minSynergy), (self.maxSynergy - self.minSynergy)), 0, 1)
        else:
            return synergy

    def invProject(self, curPos, normalize_synergy=False):
        '''Projects a transform from teleoperation space to joint space
        curPos: the teleoperation position for a given structure with shape (1x3)
        normalize_synergy: boolean if the input synergy is normalized (True) or not (False)
        '''
        if normalize_synergy:
            curPos = curPos * (self.maxSynergy - self.minSynergy) + self.minSynergy

        A = self.df[['alpha_norm', 'sigma_norm', 'epsilon_norm']].values
        offset = self.df['offset'].values

        resPos = np.dot(np.multiply(curPos, self.invScale), np.transpose(A)) + offset
        resPos = pd.Series(resPos.reshape((-1,)), self.df.index)
        return self.denormalize(resPos)

    def normalize(self, curPos:pd.Series):
        '''Normalizes the position between the min and max angles
        curPos: the joint angle for a given structure with shape [1xn]
        '''
        minSeries = self.df['min']
        maxSeries = self.df['max']
        resPos = curPos.copy()
        for ind in curPos.index:
            resPos[ind] = (curPos[ind] - minSeries[ind]) / (maxSeries[ind] - minSeries[ind])
            if resPos[ind] > 1:
                resPos[ind] = 1
            elif resPos[ind] < 0:
                resPos[ind] = 0
        return resPos

    def denormalize(self, curPos:pd.Series):
        '''Reverts a normalized position back to it's original range
        curPos: the joint angle for a given structure with shape [1xn]
        '''
        minSeries = self.df['min']
        maxSeries = self.df['max']
        resPos = curPos.copy()
        for ind in curPos.index:
            resPos[ind] = curPos[ind] * (maxSeries[ind] - minSeries[ind]) + minSeries[ind]
        return resPos

def _scaling_apply(maxVal, minVal):
    return 1
    '''
    rangeVal = np.abs(maxVal) + np.abs(minVal)
    if rangeVal == 0:
        return 0
    else:
        return 1 / rangeVal
    '''
        

def offsetFitting(meeker1: MeekerObject, meeker2: MeekerObject, df1, df2):
    '''Used to determine the minimal offset required for similar position between two hand structures to have the same.
    meeker1: The baseline structure that is assumed to have the correct synergy values
    meeker2: The structure whos synergy must be adjusted to match the first structure
    df1: Positions for the first hand structure
    df2: Positions for the second hand structure that correspond to the previous positions

    Returns the offset values for the second hand structures required to match the synergy

    *WARNING*: This function will adjust the value inside the meeker objects while running
    '''
    x0 = meeker2.df['offset']
    def eval(newOffset):
        #offError = mse(x0, newOffset, squared=True)
        offError = 0
        synError = 0
        meeker2.df['offset'] = newOffset
        for d1, d2 in zip(df1.iterrows(), df2.iterrows()):
            synergy1 = meeker1.forwardProject(d1[1])
            synergy2 = meeker2.forwardProject(d2[1])
            synError += mse(synergy1, synergy2, squared=True)
        return offError, synError
    
    res = minimize(lambda x: np.sum(eval(x)), x0)

    if res.success:
        error = eval(res.x)
        meeker2.df['offset'] = x0
        return res.x, error
    else:
        meeker2.df['offset'] = x0
        raise ValueError(res.message)

if __name__ == '__main__':
    m1 = MeekerObject(shadowHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
    m2 = MeekerObject(humanStruct)

    df1 = pd.read_excel('Synergy.xlsx', 'Robot', index_col=0)
    df2 = pd.read_excel('Synergy.xlsx', 'Human', index_col=0)

    for d1, d2 in zip(df1.iterrows(), df2.iterrows()):
        synergy1 = m1.forwardProject(d1[1])
        synergy2 = m2.forwardProject(d2[1])
        print('Original')
        print("Robot Name: ", d1[0], '\tHuman Name: ', d2[0])
        print("Robot Example: ", synergy1)
        print("Human Example: ", synergy2)
        print()

    offset, e = offsetFitting(m1, m2, df1, df2)
    print("Best offset found: ", offset)
    print("Associated error value is: ", e[1])
    print()

    for d1, d2 in zip(df1.iterrows(), df2.iterrows()):
        synergy1 = m1.forwardProject(d1[1])
        synergy2 = m2.forwardProject(d2[1])
        print("Robot Name: ", d1[0], '\tHuman Name: ', d2[0])
        print("Robot Example: ", synergy1)
        print("Human Example: ", synergy2)
        print()
