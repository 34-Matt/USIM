from constants import STRUCT_LIST
from main import USIM, train_step
from dataManager import SynergyDataManager
from modelLayers import EncoderModel, AngleDecoderModel, imageSize, latentSize, jointCounts

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError #, CosineSimilarity
from tensorflow.keras.metrics import RootMeanSquaredError

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read Arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Path to the data folder')
parser.add_argument('structs', nargs='+', help='List of structures to train on')
parser.add_argument('--save_path', help='Path to save the model', default='.')
parser.add_argument('--model_name', help='Name of the model', default='model')
parser.add_argument('--epochs', '-e', help='Number of epochs to train', type=int, default=200)
parser.add_argument('--batch', '-b', help='Batch size', default=15)
args = parser.parse_args()

structs = args.structs
dataPath = args.data_path
epoch = args.epochs
batch = args.batch
save_path = args.save_path
model_name = args.model_name

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

# Load data
trainingObjects = {}

for s in key_list:
    trainingObjects[s] = SynergyDataManager(imageSize, os.path.join(dataPath, s, 'training'), STRUCT_LIST[s])

# Training Loop
model.initialize()
optimizer = Adam(0.0005)
optimizer.build(model.getAllTrainable())
loss_fn = MeanSquaredError()
#loss_fn2 = CosineSimilarity()
loss_fn2 = None
metric_fn = RootMeanSquaredError()
alpha = 0.9
metric = {k:[] for k in key_list}
for e in range(epoch):
    for key in key_list:
        batch_count = 0
        print(f"Starting Epoch {e} on set {key}")
        for df, syn, img, _ in trainingObjects[key].randomSample(batch, True):
            # Convert df to matrix
            joint = tf.convert_to_tensor(df.values)
            synergy = tf.convert_to_tensor(syn.values, dtype=tf.float32)
            weight = (3 - np.sum(syn.values, axis=1)) / 3 + 0.01

            #Train on Data
            loss = train_step(model, key, img, [joint, synergy], optimizer, loss_fn, loss_fn2, alpha = alpha, metric_fn=metric_fn, weight=weight)
            #print(f"Trained batch {batch_count} with a loss of {loss}")
            batch_count += 1
            #hist = model.train_on_batch(image, [joint, synergy], return_dict=True)
        metric[key].append(metric_fn.result())
        metric_fn.reset_state()
    tf.keras.backend.clear_session()

# Load data
testingObject = None

## Create Save location if it doesn't exist
if not os.path.exists(save_path):
    os.mkdir(save_path)

for s in key_list:
    testingObject = SynergyDataManager(imageSize, os.path.join(dataPath, s, 'testing'), STRUCT_LIST[s])

    ## Iterate over each
    synergy_list = []
    joint_list = []
    for index in range(testingObject.sample_count):
        joint, synergy, img, = testingObject.getSample([index])
        jcols = joint.columns
        joint = np.reshape(joint.values, (1, -1))

        scols = synergy.columns
        synergy = np.reshape(synergy.values, (1, -1))
        image = img.astype(np.float64)

        predict = model(image, decoder_name=s)
        joint_pred = pd.Series(np.reshape(predict[0], (-1,)), jcols)
        joint_list.append(joint_pred)
        synergy_pred = pd.Series(np.reshape(predict[1], (-1,)), scols)
        synergy_list.append(synergy_pred)
        

    synergy_df = pd.concat(synergy_list, axis=1,ignore_index=True)
    synergy_df.T.to_csv(os.path.join(save_path, f"allTogether_{s}_encoder.csv"))

    joint_df = pd.concat(joint_list, axis=1,ignore_index=True)
    joint_df = joint_df.T.apply(testingObject.meeker.denormalize, axis=1)
    joint_df.to_csv(os.path.join(save_path, f"allTogether_{s}_decoder.csv"))

# Plot Training Progress
t_pre = np.arange(0, epoch)

plt.figure()
plt.axes()
for key in key_list:
    plt.plot(t_pre, metric[key], label=key)
plt.legend()
plt.savefig(os.path.join(save_path, 'model_loss.png'))

# Save Model
model.save(os.path.join(save_path, model_name),save_format='tf')
