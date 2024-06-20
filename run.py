from main import USIM, train_step

from dataManipulation.MymeekerStructure import MeekerObject, humanHandStruct, shadowHandStruct, shadow4HandStruct, shadow3HandStruct, allegroHandStruct
from modelLayers import EncoderModel, AngleDecoderModel, imageSize, latentSize, jointCounts

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

import os
import cv2
import glob
import random
import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main(structs):
    # Create the Model
    model = USIM(imageSize, latentSize, EncoderModel())

    key_list = []
    for s in structs:
        if 'human' in s.lower():
            model.add_decoder('Human', AngleDecoderModel(jointCounts['Human_Angle']))
            key_list.append('Human')
        elif 'shadow5' in s.lower():
            model.add_decoder('Shadow5', AngleDecoderModel(jointCounts['Shadow5']))
            key_list.append('Shadow5')
        elif 'shadow4' in s.lower():
            model.add_decoder('Shadow4', AngleDecoderModel(jointCounts['Shadow4']))
            key_list.append('Shadow4')
        elif 'shadow3' in s.lower():
            model.add_decoder('Shadow3', AngleDecoderModel(jointCounts['Shadow3']))
            key_list.append('Shadow3')

#    model.compile(
#        loss = 'mse',
#        metrics = ['mse'],
#        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#    )

    # Load data
    dataPath = os.path.join("..", "NewShadowTraining")
    trainingDf = {}
    trainingDf_synergy = {}
    meeker = {}
    img_parts = {}

    for s in key_list:
        dataPath_s = os.path.join(dataPath, s, 'training')

        ## Find CSV
        csvList = glob.glob(os.path.join(dataPath_s, '*.csv'))
        if len(csvList) != 1:
            raise FileExistsError(f"Was expecting 1 csv file in {dataPath_s}, but found {len(csvList)}.")
        trainingDf[s] = pd.read_csv(csvList[0], index_col=0)

        if 'rh_WRJ1' in trainingDf[s].columns:
            trainingDf[s] = trainingDf[s].drop(columns=['rh_WRJ1'])
        if 'rh_WRJ2' in trainingDf[s].columns:
            trainingDf[s] = trainingDf[s].drop(columns=['rh_WRJ2'])

        # Get Synergy
        if 'Human' in s:
            meeker[s] = MeekerObject(humanHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
        elif 'Shadow5' in s:
            meeker[s] = MeekerObject(shadowHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
        elif 'Shadow4' in s:
            meeker[s] = MeekerObject(shadow4HandStruct)
        elif 'Shadow3' in s:
            meeker[s] = MeekerObject(shadow3HandStruct)

        trainingDf_synergy[s] = trainingDf[s].apply(meeker[s].forwardProject, axis=1)

        ## Get list of images
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
        img_parts[s] = (img_start, img_end, img_number_length)

        ## Check that number of images match length of csv file
        if len(trainingDf[s]) != len(imgList):
            raise IndexError(f"Expected sizes to be the same, but got {trainingDf[s].shape} for csv shape and {len(imgList)} for number of images.")

    # Batch Generator
    def BatchGenerator(
            batchSize,
            key,
            trainingDf,
            trainingDf_synergy,
            meekers,
            img_parts
        ):
        totalIndexList = list(range(len(trainingDf[key])))
        random.shuffle(totalIndexList)

        while len(totalIndexList) >= batchSize:
            batch_index = totalIndexList[-batchSize:]
            del totalIndexList[-batchSize:]
            
            returndf = trainingDf[key].iloc[batch_index]
            returndf_synergy = trainingDf_synergy[key].iloc[batch_index]
            returndf = returndf.apply(meekers[key].normalize, axis=1)
            
            returnimgList = []
            for num in batch_index:
                img_start, img_end, img_number_length = img_parts[key]
                returnimgList.append(img_start+str(num).zfill(img_number_length) +img_end)

            yield returndf, returnimgList, returndf_synergy

    # Training Loop
    MAXEPOCHS_PRE = 200
    MAXBATCH_PRE = 15

    model.initialize()
    optimizer = Adam(0.0005)
    optimizer.build(model.getAllTrainable())
    loss_fn = MeanSquaredError()
    metric_fn = RootMeanSquaredError() 
    metric = {k:[] for k in key_list}
    for epoch in range(MAXEPOCHS_PRE):
        count = 0
        for key in key_list:
            print(f"Starting Epoch {epoch} on set {key}")
            for df, img, syn in BatchGenerator(MAXBATCH_PRE, key, trainingDf, trainingDf_synergy, meeker, img_parts):
                # Convert df to matrix
                joint = tf.convert_to_tensor(df.values)
                synergy = tf.convert_to_tensor(syn.values)
                weight = (3 - np.sum(syn.values, axis=1)) / 3 + 0.01

                # Load image
                image = np.zeros((MAXBATCH_PRE, 224, 224, 1))

                for batch_num in range(MAXBATCH_PRE):
                    im =cv2.imread(img[batch_num], cv2.IMREAD_UNCHANGED)
                    im = cv2.resize(im,(224, 224))
                    image[batch_num,:,:,0] = im

                #Train on Data
                train_step(model, key, image, [joint, synergy], optimizer, loss_fn, metric_fn, weight)
                #hist = model.train_on_batch(image, [joint, synergy], return_dict=True)
            metric[key].append(metric_fn.result())
            metric_fn.reset_state()
        tf.keras.backend.clear_session()

    # Load data
    dataPath = os.path.join("..", "NewShadowTraining")
    trainingDf = {}
    trainingDf_synergy = {}
    meeker = {}
    img_parts = {}

    for s in key_list:
        dataPath_s = os.path.join(dataPath, s, 'testing')

        ## Find CSV
        csvList = glob.glob(os.path.join(dataPath_s, '*.csv'))
        if len(csvList) != 1:
            raise FileExistsError(f"Was expecting 1 csv file in {dataPath_s}, but found {len(csvList)}.")
        trainingDf = pd.read_csv(csvList[0], index_col=0)

        if 'rh_WRJ1' in trainingDf.columns:
            trainingDf = trainingDf.drop(columns=['rh_WRJ1'])
        if 'rh_WRJ2' in trainingDf.columns:
            trainingDf = trainingDf.drop(columns=['rh_WRJ2'])

        # Get Synergy
        if 'Human' in s:
            meeker = MeekerObject(humanHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
        elif 'Shadow5' in s:
            meeker = MeekerObject(shadowHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
        elif 'Shadow4' in s:
            meeker = MeekerObject(shadow4HandStruct)
        elif 'Shadow3' in s:
            meeker = MeekerObject(shadow3HandStruct)

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
        synergy_list = []
        joint_list = []
        for index in range(len(imgList)):
            joint = trainingDf.iloc[index]
            jcols = joint.index
            joint = np.reshape(joint.values, (1, -1))

            synergy = trainingDf_synergy.iloc[index]
            scols = synergy.index
            synergy = np.reshape(synergy.values, (1, -1))

            image =cv2.imread(img_start+str(index).zfill(img_number_length) +img_end, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (224, 224))
            image = np.reshape(image, (1, 224, 224, 1))
            image = image.astype(np.float64)

            predict = model(image, s)
            joint_pred = pd.Series(np.reshape(predict[0], (-1,)), jcols)
            joint_list.append(joint_pred)
            synergy_pred = pd.Series(np.reshape(predict[1], (-1,)), scols)
            synergy_list.append(synergy_pred)
            

        synergy_df = pd.concat(synergy_list, axis=1,ignore_index=True)
        synergy_df.T.to_csv(f"allTogether_{s}_encoder.csv")

        joint_df = pd.concat(joint_list, axis=1,ignore_index=True)
        joint_df = joint_df.T.apply(meeker.denormalize, axis=1)
        joint_df.to_csv(f"allTogether_{s}_decoder.csv")

    # Plot Training Progress
    t_pre = np.arange(0, MAXEPOCHS_PRE)

    fig = plt.figure()
    ax = plt.axes()
    for key in key_list:
        plt.plot(t_pre, metric[key], label=key)
    plt.legend()
    plt.savefig('allTogether.png')

    # Save Model
    model.save('model',save_format='tf')

main(['Shadow5', 'Shadow4', 'Shadow3', 'Human'])