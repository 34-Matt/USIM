from MymeekerStructure import *
import pandas as pd
import numpy as np
import argparse
from typing import Union

parser = argparse.ArgumentParser(description="Evaluate the performance of Meeker's algorithm")
parser.add_argument('data', type=str, help='path to the data csv file')
parser.add_argument('struct', type=str, help='the structure of the hand [human, allegro, shadow5, shadow4, shadow3]')
parser.add_argument('--convert', type=str, default=None, help='convert the data to a different structure [human, allegro, shadow5, shadow4, shadow3]')

args = parser.parse_args()

def getStruct(structName:str) -> MeekerObject:
    if structName == 'human':
        return MeekerObject(humanHandStruct)
    elif structName == 'allegro':
        return MeekerObject(allegroHandStruct)
    elif structName == 'shadow5':
        return MeekerObject(shadowHandStruct)
    elif structName == 'shadow4':
        return MeekerObject(shadow4HandStruct)
    elif structName == 'shadow3':
        return MeekerObject(shadow3HandStruct)
    else:
        raise ValueError('Unknown structure name', structName)
    
def convertData(data:pd.DataFrame, dataStruct:MeekerObject, desiredStruct:MeekerObject) -> pd.DataFrame:
    '''Converts the data from one structure to another
    
    Args:
        data (pd.DataFrame): the data to convert
        dataStruct (MeekerObject): the structure of the data
        desiredStruct (MeekerObject): the structure to convert the data to

    Returns:
        convertedData (pd.DataFrame): the converted data
    '''
    convertedData = pd.DataFrame(columns=desiredStruct.joint_col)
    for index, row in data.iterrows():
        convertedData = convertedData.append(desiredStruct.invProject(dataStruct.forwardProject(row)), ignore_index=True)

    return convertedData

def evaluate(data:pd.DataFrame, dataStruct:MeekerObject) -> Union[float, int, pd.Series]:
    '''Evaluates the performance of the algorithm with rmse
    
    Args:
        data (pd.DataFrame): the data to evaluate
        dataStruct (MeekerObject): the structure of the data
        
        Returns:
            rmse (float): the rmse of the data
            index (int): the index of the data with the absolute error
            predicted (pd.Series): the predicted series of the data with the largest error

    '''
    convertedData = convertData(data, dataStruct, dataStruct)
    error2 = (data - convertedData)**2
    error2 = error2.mean(axis=1)
    error = np.sqrt(error2)

    rmse = np.mean(error)
    index = np.argmax(error)
    predicted = convertedData.iloc[index]

    return rmse, index, predicted

def main():
    data = pd.read_csv(args.data, index_col=0)
    dataStruct = getStruct(args.struct)
    if args.convert is None:
        rmse, ind, pred = evaluate(data, dataStruct)
        print('Converting back to self, the average rmse is', rmse)
        print('The index of the largest error is', ind)
        print(pred)
        print(data.iloc[ind])
    else:
        desiredStruct = getStruct(args.convert)
        convertedData = convertData(data, dataStruct, desiredStruct)
        convertedData.to_csv(args.data[:-4] + '_' + args.convert + '_converted.csv')

if __name__ == '__main__':
    main()
