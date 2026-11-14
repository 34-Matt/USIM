import os
import pickle
import argparse
import numpy as np
import pandas as pd
from constants import *
import tensorflow as tf
import matplotlib.pyplot as plt

from dataManager import SynergyDataManager
from modelLayers import EncoderModel, AngleDecoderModel, imageSize, jointCounts

from typing import List, Any, Union, Tuple

class TeachNet(tf.keras.Model):
    def __init__(
            self,
            encoder1: tf.keras.Model,
            decoder1: tf.keras.Model,
            encoder2: tf.keras.Model,
            decoder2: tf.keras.Model,
            input_size:List[int] = None,
            **kwargs
    ):
        '''Creates the TeachNet Model
        
        Args:
            encoder1 (List[Layer]): The encoder model for the primary model
            decoder1 (List[Layer]): The decoder model for the primary model
            encoder2 (List[Layer]): The encoder model for the secondary model
            decoder2 (List[Layer]): The decoder model for the secondary model
            input_size (List[int]): The input size of the model (for optional initializing)
        '''
        super().__init__(**kwargs)
        self.encoder1 = encoder1
        self.decoder1 = decoder1
        self.encoder2 = encoder2
        self.decoder2 = decoder2

        # Initialize
        if input_size is not None:
            temp_input = tf.ones((5, *input_size))
            temp_latent = self.encoder2(temp_input)
            self.decoder2(temp_latent)

            temp_latent = self.encoder1(temp_input)
            self.decoder1(temp_latent)
            self.decoder2(temp_latent)

    def call_primary(
            self,
            inputs:tf.Tensor,
            training=False,
            mask=None
    ) -> List[tf.Tensor]:
        '''Calls the primary model
        
        Args:
            inputs [Tensor]: The input to the model
            training [bool]: If the model is training or not
            mask [Tensor]: The mask to apply to the model
        
        Returns:
            Prediction [Tensor]: The output of the model
            Latent [Tensor]: The resulting latent features
        '''
        Latent = self.encoder1.call(inputs, training=training)
        Prediction = self.decoder1.call(Latent, training=training)
        return [Prediction, Latent]
    
    def call_secondary(
            self,
            inputs:tf.Tensor,
            training=False,
            mask=None
    ) -> List[tf.Tensor]:
        '''Calls the secondary model
        
        Args:
            inputs [Tensor]: The input to the model
            training [bool]: If the model is training or not
            mask [Tensor]: The mask to apply to the model
        
        Returns:
            Prediction [Tensor]: The output of the model
            Latent [Tensor]: The resulting latent features
        '''
        Latent = self.encoder2.call(inputs, training=training)
        Prediction = self.decoder2.call(Latent, training=training)
        return [Prediction, Latent]
    
    def getPrimaryTrainable(self) -> list:
        '''Gets all trainable weights in the primary model
        
        Returns:
            weights [list]: The list of weights in the USIM model
        '''
        weights = self.encoder1.trainable_weights
        weights.extend(self.decoder1.trainable_weights)
        return weights
    
    def getSecondaryTrainable(self) -> list:
        '''Gets all trainable weights in the secondary model
        
        Returns:
            weights [list]: The list of weights in the USIM model
        '''
        weights = self.encoder2.trainable_weights
        weights.extend(self.decoder2.trainable_weights)
        return weights

    def save(
            self,
            filepath: Any,
            overwrite: bool = True,
            include_optimizer: bool = True,
            save_format: Any = None,
            signatures: Any = None,
            options: Any = None,
            save_traces: bool = True
        ) -> None:
        '''Saves the model
        
        Args:
            filepath [Any]: The location to save the model
            overwrite [bool]: If the model should overwrite the file
            include_optimizer [bool]: If the optimizer should be included in the save
            save_format [Any]: The format to save the model in
            signatures [Any]: The signatures to save with the model
            options [Any]: The options to save with the model
            save_traces [bool]: If the traces should be saved with the model
        '''
        self.encoder1.save(
            os.path.join(filepath, 'primary_encoder'),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces
            )
        self.decoder1.save(
            os.path.join(filepath, 'primary_decoder'),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces
            )
        
        self.encoder2.save(
            os.path.join(filepath, 'secondary_encoder'),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces
            )
        self.decoder2.save(
            os.path.join(filepath, 'secondary_decoder'),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces
            )

    @staticmethod
    def load(filepath:str):
        '''Loads a model from a file

        Args:
            filepath [str]: The location of the model to load
        '''
        encoder1 = tf.keras.models.load_model(os.path.join(filepath, 'primary_encoder'))
        decoder1 = tf.keras.models.load_model(os.path.join(filepath, 'primary_decoder'))

        encoder2 = tf.keras.models.load_model(os.path.join(filepath, 'secondary_encoder'))
        decoder2 = tf.keras.models.load_model(os.path.join(filepath, 'secondary_decoder'))


        model = TeachNet(encoder1=encoder1, decoder1=decoder1, encoder2=encoder2, decoder2=decoder2)
        return model


@tf.function
def train_step_primary(
    model: "TeachNet",
    x: tf.Tensor,
    y: tf.Tensor,
    optimizer: Union[str, tf.keras.optimizers.Optimizer],
    loss_fn: Union[str, tf.keras.losses.Loss],
    metric_fn: tf.keras.metrics.Metric = None,
    weight = None
) -> float:
    '''Trains a Teachnet model on a batch of data
    
    Args:
        model [USIM]: The USIM model to train on
        x [Tensor]: The input to the USIM model
        y [Tensor]: The expected output of the USIM model
        optimizer [Union[str, Optimizaer]]: The optimizer for training the model
        loss_fn [Union[str, Loss]]: The loss function to implement
        metric_fn [Metric]: The evaluation metric (default None)

    Returns:
        loss [float]: The loss value from the training
    '''
    # Get optimizer and loss functions if a string
    if isinstance(optimizer, str):
        optimizer = tf.keras.optimizers.get(optimizer)
    
    if isinstance(loss_fn, str):
        loss_fn = tf.keras.losses.get(loss_fn)

    # Begin tracking gradient
    with tf.GradientTape() as tape:
        logits = model.call_primary(x, training=True)
        loss = loss_fn(y, logits[0], sample_weight=weight)

    # Apply the gradient:
    weights = model.getPrimaryTrainable()
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))

    # Update metric if applicable
    if metric_fn is not None:
        metric_fn.update_state(y, logits[0])

    return loss

@tf.function
def train_step_secondary(
    model: "TeachNet",
    x: Tuple[tf.Tensor, tf.Tensor],
    y: Tuple[tf.Tensor, tf.Tensor],
    optimizer: Union[str, tf.keras.optimizers.Optimizer],
    loss_fn: Union[str, tf.keras.losses.Loss],
    metric_fn: tf.keras.metrics.Metric = None,
    weight = None
) -> float:
    '''Trains a Teachnet model on a batch of data
    
    Args:
        model [USIM]: The USIM model to train on
        x [Tuple[Tensor, Tensor]]: The input to the USIM model, first for primary then for secondary
        y [Tuple[Tensor, Tensor]]: The expected output of the USIM model, first for primary then for secondary
        optimizer [Union[str, Optimizaer]]: The optimizer for training the model
        loss_fn [Union[str, Loss]]: The loss function to implement
        metric_fn [Metric]: The evaluation metric (default None)

    Returns:
        loss [float]: The loss value from the training
    '''
    # Get optimizer and loss functions if a string
    if isinstance(optimizer, str):
        optimizer = tf.keras.optimizers.get(optimizer)
    
    if isinstance(loss_fn, str):
        loss_fn = tf.keras.losses.get(loss_fn)

    # Begin tracking gradient
    with tf.GradientTape() as tape:
        logits1 = model.call_primary(x[0], training=False)
        logits2 = model.call_secondary(x[1], training=True)
        loss = loss_fn(y[1], logits2[0], sample_weight=weight)
        loss += loss_fn(logits1[1], logits2[1], sample_weight=weight)

    # Apply the gradient:
    weights = model.getSecondaryTrainable()
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))

    # Update metric if applicable
    if metric_fn is not None:
        metric_fn.update_state(y[1], logits2[0])

    return loss


def main():

    # Read Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to the data folder with paired data')
    parser.add_argument('input', type=str, help='Structures to train as input')
    parser.add_argument('output', type=str, help='Structures to train as output')
    parser.add_argument('latent', type=int, help='The number of latent features from the encoder')
    parser.add_argument('--save_path', help='Path to save the model', default='.')
    parser.add_argument('--model_name', help='Name of the model', default='model')
    parser.add_argument('--epoch1',type=int, help='Number of epochs to train the primary model', default=800)
    parser.add_argument('--epoch2',type=int, help='Number of epochs to train the secondary model', default=800)
    parser.add_argument('--batch', '-b', type=int, help='Batch size', default=15)
    args = parser.parse_args()

    dataPath = args.data_path
    latentSize = args.latent
    batch = args.batch
    save_path = args.save_path
    model_name = args.model_name


    # Create the Model
    print("Creating Models")
    def getDecoder(struct):
        if 'human' in struct:
            return AngleDecoderModel(jointCounts['Human_Angle']), "Human"
        elif 'shadow5' in struct:
            return AngleDecoderModel(jointCounts['Shadow5']), "Shadow5"
        elif 'shadow4' in struct:
            return AngleDecoderModel(jointCounts['Shadow4']), "Shadow4"
        elif 'shadow3' in struct:
            return AngleDecoderModel(jointCounts['Shadow3']), "Shadow3"
        elif 'allegro' in struct:
            return AngleDecoderModel(jointCounts['Allegro']), "Allegro"
        else:
            raise ValueError(f"Invalid structure: {struct}")
        
    decoder1, primary_key = getDecoder(args.input.lower())
    decoder2, secondary_key = getDecoder(args.output.lower())
    model = TeachNet(
        EncoderModel(latentSize=latentSize),
        decoder1,
        EncoderModel(latentSize=latentSize),
        decoder2,
        imageSize
        )

    # Load data
    print("Loading Data")
    InputData = SynergyDataManager(imageSize, os.path.join(dataPath, primary_key), STRUCT_LIST[primary_key], BAD_JOINTS[primary_key])
    OutputData = SynergyDataManager(imageSize, os.path.join(dataPath, secondary_key), STRUCT_LIST[secondary_key], BAD_JOINTS[secondary_key])

    # Training Loop
    optimizer = tf.keras.optimizers.Adam(0.0001)
    optimizer.build(model.getPrimaryTrainable())
    loss_fn = tf.keras.losses.MeanSquaredError()
    metric_fn = tf.keras.metrics.RootMeanSquaredError() 
    metric = [[], []]
    print("Beginning Training Loop")
    for ep in range(args.epoch1):
        for df, syn, img, _ in InputData.randomSample(batch):
            # Convert df to matrix
            joint = tf.convert_to_tensor(df.values)
            weight = (3 - np.sum(syn.values, axis=1)) / 3 + 0.01

            #Train on Data
            train_step_primary(model, img, joint, optimizer, loss_fn, metric_fn, weight)
        metric[0].append(metric_fn.result())
        metric_fn.reset_state()
        tf.keras.backend.clear_session()
    
    print("Beginning Second Training Loop")
    optimizer = tf.keras.optimizers.Adam(0.0001)
    optimizer.build(model.getSecondaryTrainable())
    for ep in range(args.epoch2):
        for input_df, syn, input_img, index in InputData.randomSample(batch):
            output_df, _, output_img = OutputData.getSample(index)
            # Convert df to matrix
            input_joint = tf.convert_to_tensor(input_df.values)
            output_joint = tf.convert_to_tensor(output_df.values)
            weight = (3 - np.sum(syn.values, axis=1)) / 3 + 0.01

            #Train on Data
            train_step_secondary(
                model,
                [input_img, output_img],
                [input_joint, output_joint],
                optimizer,
                loss_fn,
                metric_fn,
                weight)
        metric[1].append(metric_fn.result())
        metric_fn.reset_state()
        tf.keras.backend.clear_session()

    #Setup saving location
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    #Load data
    jcols1 = InputData.jointDf.columns
    joint1_list = []
    jcols2 = OutputData.jointDf.columns
    joint2_list = []

    for index in range(InputData.sample_count):
        _, _, img, = InputData.getSample([index])
        image = img.astype(np.float64)

        predict = model.call_primary(image)
        joint_pred = pd.Series(np.reshape(predict[0], (-1,)), jcols1)
        joint1_list.append(joint_pred)

    for index in range(OutputData.sample_count):
        _, _, img, = OutputData.getSample([index])
        image = img.astype(np.float64)

        predict = model.call_secondary(image)
        joint_pred = pd.Series(np.reshape(predict[0], (-1,)), jcols2)
        joint2_list.append(joint_pred)
        

    joint_df = pd.concat(joint1_list, axis=1,ignore_index=True)
    joint_df = joint_df.T.apply(InputData.meeker.denormalize, axis=1)
    joint_df.to_csv(os.path.join(save_path, f"primary_{args.input}.csv"))

    joint_df = pd.concat(joint2_list, axis=1,ignore_index=True)
    joint_df = joint_df.T.apply(OutputData.meeker.denormalize, axis=1)
    joint_df.to_csv(os.path.join(save_path, f"secondary_{args.output}.csv"))

    # Plot Training Progress
    t_pre = np.arange(0, args.epoch1)
    t_post = np.arange(args.epoch1, args.epoch1+args.epoch2)

    plt.figure()
    plt.axes()
    plt.plot(t_pre, metric[0], label="Teacher")
    plt.plot(t_post, metric[1], label="Student")
    plt.legend()
    plt.savefig(os.path.join(save_path, 'model_loss.png'))

    with open(os.path.join(save_path, "save_metric.csv"), "w") as f:
        for metric_count in range(len(metric)):
            f.write(",".join(str(m.numpy().item()) for m in metric[metric_count]))
            if metric_count != len(metric)-1:
                f.write(";")

    # Save Model
    model.save(os.path.join(save_path, model_name),save_format='tf')

if __name__ == "__main__":
    main()