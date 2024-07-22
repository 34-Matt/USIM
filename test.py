import os
import cv2
import glob
import argparse
import regex as re
import numpy as np
import pandas as pd

from main import USIM
from dataManager import SynergyDataManager
from modelLayers import imageSize
from constants import STRUCT_LIST

from dataManipulation.MymeekerStructure import MeekerObject, SYNERGY_COL, humanHandStruct, shadowHandStruct, shadow4HandStruct, shadow3HandStruct, allegroHandStruct


# Read Arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Path to the data folder')
parser.add_argument('model_path', help='Path to the model folder')
parser.add_argument('model_name', help='Name of the model')
parser.add_argument('input_struct', help='Input structure of the model')
args = parser.parse_args()

dataPath = args.data_path
modelPath = args.model_path
modelName = args.model_name
inputStruct = args.input_struct

# Create the Model
model = USIM.load(os.path.join(modelPath, modelName))

# Load data
trainingObjects = SynergyDataManager(imageSize, os.path.join(dataPath, inputStruct, 'testing'), STRUCT_LIST[inputStruct])

## Iterate over each
for model_struct in model.decoder.keys():
    synergy_list = []
    joint_list = []
    if 'human' in model_struct.lower():
        meeker_struct = MeekerObject(humanHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
    elif 'shadow5' in model_struct.lower():
        meeker_struct = MeekerObject(shadowHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
    elif 'shadow4' in model_struct.lower():
        meeker_struct = MeekerObject(shadow4HandStruct)
    elif 'shadow3' in model_struct.lower():
        meeker_struct = MeekerObject(shadow3HandStruct)

    jcols = meeker_struct.joint_col

    for index in range(trainingObjects.sample_count):
        joint, synergy, image = trainingObjects.getSample([index])

        predict = model(image, decoder_name=model_struct)

        joint_pred = pd.Series(np.reshape(predict[0], (-1,)), jcols)
        joint_list.append(joint_pred)
        synergy_pred = pd.Series(np.reshape(predict[1], (-1,)), SYNERGY_COL)
        synergy_list.append(synergy_pred)
        

    synergy_df = pd.concat(synergy_list, axis=1,ignore_index=True)
    synergy_df.T.to_csv(os.path.join(modelPath, f"posttesting_{inputStruct}In_{model_struct}Out_encoder.csv"))

    joint_df = pd.concat(joint_list, axis=1,ignore_index=True)
    joint_df = joint_df.T.apply(meeker_struct.denormalize, axis=1)
    joint_df.to_csv(os.path.join(modelPath, f"posttesting_{inputStruct}In_{model_struct}Out_decoder.csv"))
