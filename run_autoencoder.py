from constants import *
from main import USIM, train_step
from dataManager import SynergyDataManager
from modelLayers import EncoderModel, AngleDecoderModel, imageSize, jointCounts

import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, \
    Dropout, GlobalAveragePooling2D, Concatenate, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Read Arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Path to the data folder with paired data')
parser.add_argument('input', type=str, help='Structures to train as input')
parser.add_argument('output', type=str, help='Structures to train as output')
parser.add_argument('latent', type=int, help='The number of latent features from the encoder')
parser.add_argument('--save_path', help='Path to save the model', default='.')
parser.add_argument('--model_name', help='Name of the model', default='model')
parser.add_argument('--epochs', '-e', type=int, help='Number of epochs to train', default=200)
parser.add_argument('--batch', '-b', type=int, help='Batch size', default=15)
args = parser.parse_args()

dataPath = args.data_path
latentSize = args.latent
epoch = args.epochs
batch = args.batch
save_path = args.save_path
model_name = args.model_name

# Create the Model
print("Creating Models")
if 'human' in args.output.lower():
    decoder = AngleDecoderModel(jointCounts['Human_Angle'])
    output_struct = "Human"
elif 'shadow5' in args.output.lower():
    decoder = AngleDecoderModel(jointCounts['Shadow5'])
    output_struct = "Shadow5"
elif 'shadow4' in args.output.lower():
    decoder = AngleDecoderModel(jointCounts['Shadow4'])
    output_struct = "Shadow4"
elif 'shadow3' in args.output.lower():
    decoder = AngleDecoderModel(jointCounts['Shadow3'])
    output_struct = "Shadow3"
elif 'allegro' in args.output.lower():
    decoder = AngleDecoderModel(jointCounts['Allegro'])
    output_struct = "Allegro"
else:
    raise ValueError(f"Invalid structure: {args.output}")

model = [EncoderModel(latentSize=latentSize), decoder]

# Load data
print("Loading Data")
trainingObjects = {}

s = args.input
InputData = SynergyDataManager(imageSize, os.path.join(dataPath, s), STRUCT_LIST[s], BAD_JOINTS[s])
s = args.output
OutputData = SynergyDataManager(imageSize, os.path.join(dataPath, s), STRUCT_LIST[s], BAD_JOINTS[s])

# Training Loop
optimizer = tf.keras.optimizers.Adam(0.00005)
trainable_weights = model[0].trainable_weights
trainable_weights.extend(model[1].trainable_weights)
optimizer.build(trainable_weights)

loss_fn = tf.keras.losses.MeanSquaredError()
metric_fn = tf.keras.metrics.RootMeanSquaredError() 
metric = []
print("Beginning Training Loop")
for ep in range(epoch):
    print("Epoch: ", ep)
    current_metrics = 0.0
    batches = 0
    for _, _, img, sample_ids in InputData.randomSample(batch):
        df, syn, _ = OutputData.getSample(sample_ids)
        weight = (3 - np.sum(syn.values, axis=1)) / 3 + 0.01

        # Convert df to matrix
        joint = tf.convert_to_tensor(df.values, dtype=tf.float32)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        weight = tf.convert_to_tensor(weight, dtype=tf.float32)

        # Train
        with tf.GradientTape() as tape:
            latent = model[0].call(img, training=True)
            logit = model[1].call(latent, training=True)
            loss = loss_fn(joint, logit, sample_weight=weight)

        # Apply the gradient:
        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))

        # Update metric if applicable
        if metric_fn is not None:
            metric_fn.update_state(joint, logit)

    metric.append(metric_fn.result())
    metric_fn.reset_state()
    tf.keras.backend.clear_session()

#Setup saving location
if not os.path.exists(save_path):
    os.mkdir(save_path)

#Load data
jcols = OutputData.jointDf.columns
joint_list = []

for index in range(InputData.sample_count):
    _, _, img, = InputData.getSample([index])
    image = img.astype(np.float64)

    latent = model[0](image)
    predict = model[1](latent)
    joint_pred = pd.Series(np.reshape(predict, (-1,)), jcols)
    joint_list.append(joint_pred)
    

joint_df = pd.concat(joint_list, axis=1,ignore_index=True)
joint_df = joint_df.T.apply(InputData.meeker.denormalize, axis=1)
joint_df.to_csv(os.path.join(save_path, f"allTogether_{s}_decoder.csv"))

# Plot Training Progress
t_pre = np.arange(0, epoch)

plt.figure()
plt.axes()
plt.plot(t_pre, metric)
plt.savefig(os.path.join(save_path, 'model_loss.png'))

with open(os.path.join(save_path, "save_metric.csv"), "w") as f:
    f.write(",".join(str(m.numpy().item()) for m in metric))

# Save Model
model[0].save(os.path.join(save_path, model_name + "_encoder"),save_format='tf')
model[1].save(os.path.join(save_path, model_name + "_decoder"),save_format='tf')
