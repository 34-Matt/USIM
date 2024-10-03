"""Folder containing the main code for the USIM model"""

import os
import pickle
import tensorflow as tf

from typing import List, Tuple, Union, Any


class USIM(tf.keras.Model):
    def __init__(
            self,
            input_size:List[int],
            latent_size: int,
            encoder_layers: tf.keras.Model,
            **kwargs
    ):
        '''Creates the USIM Model
        
        Args:
            input_size (List[int]): The input size of the model
            latent_size (int): The size of the latent features in the model
            encoder_layers (List[Layer]): The list of layers for the encoder portion of the model
        '''
        super().__init__(**kwargs)
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = encoder_layers
        self.decoder = {}

    def validateInput(
            self,
            inputs: tf.Tensor,
            verbose:int = 0
    ) -> bool:
        '''Validates the input
        
        Args:
            inputs (Tensor): The input to the system, expected to have size of (N, *input_size)
            verbose (int): The level of information to display (0=None, 1=Only Error, 2=All Info)

        Returns:
            valid (bool): A boolean stating if the input is valid (True) or not (False)
        '''
        shape = tf.shape(inputs)

        # General Info
        if verbose >= 2:
            print(f'Expected Shape:\n\tValue: {self.input_size}\n\tDim: {len(self.input_size)}')
            print(f'Input Shape:\n\tValue: {shape}\n\tDim: {len(shape)}\n\t Input Count: {shape[0]}\n\t Input Value: {shape[1:]}')

        # Check Dimension
        if len(shape) != len(self.input_size)+1:
            if verbose >= 1:
                print(f'Input shape {shape} is expected to have one more dimention then {self.input_size}')
            return False
        
        # Check Value
        for i, size_i in enumerate(self.input_size):
            if shape[i+1] != size_i:
                if verbose >= 1:
                    print(f'Input shape {shape} is expected to have a size of {size_i} at index {i+1}')
                return False
        
        if verbose >= 2:
            print('Found nothing wrong, input is valid')
        return True

    def add_decoder(
            self,
            name: str,
            layers: tf.keras.Model
    ) -> None:
        '''Adds a decoder to the model
        
        Args:
            name (str): A name to identify the decoder
            layers (List[Layer]): The list of layers for the decoder portion of the model
        '''
        if name in self.decoder:
            raise IndexError(f"The name {name} already exists as the decoder in the model.")

        self.decoder[name] = layers

    def call(
            self,
            inputs:tf.Tensor,
            training=False,
            mask=None,
            decoder_name:str=None
    ) -> List[tf.Tensor]:
        '''Calls the model
        
        Args:
            inputs [Tensor]: The input to the model
            training [bool]: If the model is training or not
            mask [Tensor]: The mask to apply to the model
            decoder_name [str]: The name of the decoder to output
        
        Returns:
            Prediction [Tensor]: The output of the model
            Latent [Tensor]: The resulting latent features
        '''
        if decoder_name is None:
            raise ValueError('Decoder name is required to call the model')
        Latent = self.encoder.call(inputs, training=training, mask=mask)
        Prediction = self.decoder[decoder_name].call(Latent, training=training)
        return [Prediction, Latent]

    def getTrainable(
            self,
            decoder_name:str
    ) -> list:
        '''Returns a list of all trainable weights in the model
        
        Args:
            decoder_name [str]: The name of the decoder being worked on
            
        Returns:
            weights [list]: The list of weights in the autoencoder process
        '''
        weights = self.encoder.trainable_weights
        weights.extend(self.decoder[decoder_name].trainable_weights)
        return weights

    def getAllTrainable(self) -> list:
        '''Gets all trainable weights in the model
        
        Returns:
            weights [list]: The list of weights in the USIM model
        '''
        weights = self.encoder.trainable_weights
        for decoder in self.decoder.values():
            weights.extend(decoder.trainable_weights)
        return weights
    
    def initialize(self):
        '''Builds the USIM model for usage.'''
        temp_input = tf.ones((5, *self.input_size))
        temp_latent = tf.ones((5, self.latent_size))

        self.encoder(temp_input)
        for decoder in self.decoder.values():
            decoder(temp_latent)

    def get_loss(self, decoder:str = None) -> float:
        '''Gets the internal losses to individual encoder and decoders'''
        loss = 0
        if decoder is None:
            for d in self.decoder:
                loss += d.get_loss()
        else:
            loss = self.decoder[decoder].get_loss()
        return loss

    #pylint: disable=too-many-arguments
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
        self.encoder.save(
            os.path.join(filepath, 'encoder'),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces
            )
        
        for name, decoder in self.decoder.items():
            decoder.save(
                os.path.join(filepath, 'decoder_'+name),
                overwrite=overwrite,
                include_optimizer=include_optimizer,
                save_format=save_format,
                signatures=signatures,
                options=options,
                save_traces=save_traces
                )
        
        with open(os.path.join(filepath, 'param.pickle'), 'wb') as f:
            pickle.dump((self.input_size, self.latent_size), f)

    #pylint: disable=too-many-arguments
    @staticmethod
    def load(filepath:str):
        '''Loads a USIM model from a file

        Args:
            filepath [str]: The location of the model to load
        '''
        with open(os.path.join(filepath, 'param.pickle'), 'rb') as f:
            input_size, latent_size = pickle.load(f)

        encoder = tf.keras.models.load_model(os.path.join(filepath, 'encoder'))
        
        model = USIM(input_size, latent_size, encoder)
        
        for file in os.listdir(filepath):
            if file.startswith('decoder_'):
                name = file[8:]
                model.add_decoder(name, tf.keras.models.load_model(os.path.join(filepath, file)))
        
        return model


@tf.function
def train_step(
    model: "USIM",
    decoder_name: str,
    x: tf.Tensor,
    y: Tuple[tf.Tensor, tf.Tensor],
    optimizer: Union[str, tf.keras.optimizers.Optimizer],
    loss_fn1: Union[str, tf.keras.losses.Loss],
    loss_fn2: Union[str, tf.keras.losses.Loss] = None,
    alpha: float = 1.0,
    metric_fn: tf.keras.metrics.Metric = None,
    weight = None
) -> float:
    '''Trains a USIM model on a batch of data
    
    Args:
        model [USIM]: The USIM model to train on
        decoder_name [str]: The name of the decoder that is associated with this batch
        x [Tensor]: The input to the USIM model
        y [Tuple[Tensor, Tensor]]: The expected output of the USIM model, with the first element being the expected prediction and the second being the expected latent features
        optimizer [Union[str, Optimizer]]: The optimizer to apply the gradient decent
        loss_fn [Union[str, Loss]]: The loss function to implement
        metric_fn [Metric]: The evaluation metric (default None)

    Returns:
        loss [float]: The loss value from the training
    '''
    # Get optimizer and loss functions if a string
    if isinstance(optimizer, str):
        optimizer = tf.keras.optimizers.get(optimizer)
    
    if isinstance(loss_fn1, str):
        loss_fn1 = tf.keras.losses.get(loss_fn1)
        
    if isinstance(loss_fn2, str):
        loss_fn2 = tf.keras.losses.get(loss_fn2)

    # Begin tracking gradient
    with tf.GradientTape() as tape:
        logits = model(x, decoder_name=decoder_name, training=True)
        
        loss = loss_fn1(y[0], logits[0], sample_weight=weight)
        if loss_fn2 is None:
            loss += tf.multiply(tf.cast(alpha, tf.float32), loss_fn1(y[1], logits[1], sample_weight=weight))
        else:
            loss +=  tf.multiply(tf.cast(alpha, tf.float32), loss_fn2(y[1], logits[1], sample_weight=weight))
        loss += model.get_loss(decoder_name)

    # Apply the gradient:
    weights = model.getTrainable(decoder_name)
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))

    # Update metric if applicable
    metric_fn.update_state(y[0], logits[0])
    metric_fn.update_state(y[1], logits[1])

    return loss
