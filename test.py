import os
import cv2
import glob
import argparse
import regex as re
import numpy as np
import pandas as pd

from main import USIM
from dataManager import SynergyDataManager

from dataManipulation.MymeekerStructure import MeekerObject, humanHandStruct, shadowHandStruct, shadow4HandStruct, shadow3HandStruct, allegroHandStruct


# Read Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='Path to the data folder')
parser.add_argument('--model_path', help='Path to the model folder')
parser.add_argument('--model_name', help='Name of the model')
parser.add_argument('--input_struct', help='Input structure of the model')
args = parser.parse_args()

dataPath = args.data_path
modelPath = args.model_path
modelName = args.model_name
inputStruct = args.input_struct

# Create the Model
model = USIM.load(os.path.join(modelPath, modelName))

# Load data
trainingObjects = {}

for s in model.decoder.keys():
    trainingObjects[s] = SynergyDataManager(imageSize, os.path.join(dataPath, s, 'testing'), STRUCT_LIST[s], BAD_JOINTS[s])
trainingDf = {}
trainingDf_synergy = {}
meeker = {}

dataPath_s = os.path.join(dataPath, inputStruct, 'testing')

## Find CSV
csvList = glob.glob(os.path.join(dataPath_s, '*.csv'))
if len(csvList) != 1:
    raise FileExistsError(f"Was expecting 1 csv file in {dataPath_s}, but found {len(csvList)}.")
trainingDf = pd.read_csv(csvList[0], index_col=0)

if 'rh_WRJ1' in trainingDf.columns:
    trainingDf = trainingDf.drop(columns=['rh_WRJ1'])
if 'rh_WRJ2' in trainingDf.columns:
    trainingDf = trainingDf.drop(columns=['rh_WRJ2'])

meeker = MeekerObject(shadowHandStruct, ['rh_WRJ1', 'rh_WRJ2'])

trainingDf_synergy = trainingDf.apply(meeker.forwardProject, axis=1)

## Get list of imagesencoder_loss.append(l)
imgList = glob.glob(os.path.join(dataPath_s, '*.png'))
tempimg = imgList[0]

number_pattern = '([0-9]+)'
for numberIndex in re.finditer(number_pattern, tempimg):
    pass
#numberIndex = next(re.finditer(number_pattern, tempimg))
number_start, number_end = numberIndex.start(0), numberIndex.end(0)

img_start = tempimg[0:number_start]
img_end = tempimg[number_end:]
img_number_length = number_end - number_start

## Check that number of images match length of csv file
if len(trainingDf) != len(imgList):
    raise IndexError(f"Expected sizes to be the same, but got {trainingDf.shape} for csv shape and {len(imgList)} for number of images.")

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

    for index in range(len(imgList)):
        joint = trainingDf.iloc[index]
        joint = np.reshape(joint.values, (1, -1))

        synergy = trainingDf_synergy.iloc[index]
        scols = synergy.index
        synergy = np.reshape(synergy.values, (1, -1))

        image =cv2.imread(img_start+str(index).zfill(img_number_length) +img_end, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (224, 224))
        image = np.reshape(image, (1, 224, 224, 1))
        image = image.astype(np.float64)

        predict = model(image, model_struct)
        joint_pred = pd.Series(np.reshape(predict[0], (-1,)), jcols)
        joint_list.append(joint_pred)
        synergy_pred = pd.Series(np.reshape(predict[1], (-1,)), scols)
        synergy_list.append(synergy_pred)
        

    synergy_df = pd.concat(synergy_list, axis=1,ignore_index=True)
    synergy_df.T.to_csv(os.path.join(modelPath, f"posttesting_{inputStruct}In_{model_struct}Out_encoder.csv"))

    joint_df = pd.concat(joint_list, axis=1,ignore_index=True)
    joint_df = joint_df.T.apply(meeker.denormalize, axis=1)
    joint_df.to_csv(os.path.join(modelPath, f"posttesting_{inputStruct}In_{model_struct}Out_decoder.csv"))
