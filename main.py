import os
import pickle
import tensorflow as tf

from typing import List, Tuple, Union


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
            input: tf.Tensor,
            verbose:int = 0
    ) -> bool:
        '''Validates the input
        
        Args:
            input (Tensor): The input to the system, expected to have size of (N, *input_size)
            verbose (int): The level of information to display (0=None, 1=Only Error, 2=All Info)

        Returns:
            valid (bool): A boolean stating if the input is valid (True) or not (False)
        '''
        shape = tf.shape(input)

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
        for i in range(len(self.input_size)):
            if self.input_size[i] != shape[i+1]:
                if verbose >= 1:
                    print(f'Found an instance where input shape differs from expectation. {shape} != {self.input_size}')
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
            decoder_layers (List[Layer]): The list of layers for the decoder portion of the model
        '''
        if name in self.decoder.keys():
            raise IndexError(f"The name {name} already exists as the decoder in the model.")

        self.decoder[name] = layers

    def call(
            self,
            input_tensor:tf.Tensor,
            decoder_name:str,
            training=False
    ) -> List[tf.Tensor]:
        '''Calls the model
        
        Args:
            input_tensor [Tensor]: The input to the model
            decoder_name [str]: The name of the decoder to output
            **kwargs: Extra key word arguments to pass through the call function
        
        Returns:
            Prediction [Tensor]: The output of the model
            Latent [Tensor]: The resulting latent features
        '''
        Latent = self.encoder.call(input_tensor, training=training)
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
        for name in self.decoder.keys():
            weights.extend(self.decoder[name].trainable_weights)
        return weights
    
    def initialize(self):
        '''Builds the USIM model for usage.'''
        temp_input = tf.ones((5, *self.input_size))
        temp_latent = tf.ones((5, self.latent_size))

        self.encoder(temp_input)
        for name in self.decoder.keys():
            self.decoder[name](temp_latent)

    def save(self, loc, *args, **kwargs):
        self.encoder.save(os.path.join(loc, 'encoder'), *args, **kwargs)
        
        for name in self.decoder.keys():
            self.decoder[name].save(os.path.join(loc, 'decoder_'+name), *args, **kwargs)
        
        with open(os.path.join(loc, 'param.pickle'), 'wb') as f:
            pickle.dump((self.input_size, self.latent_size), f)

    @staticmethod
    def load(loc, *args, **kwargs):
        with open(os.path.join(loc, 'param.pickle'), 'rb') as f:
            input_size, latent_size = pickle.load(f)

        encoder = tf.keras.models.load_model(os.path.join(loc, 'encoder'))
        
        model = USIM(input_size, latent_size, encoder)
        
        for file in os.listdir(loc):
            if file.startswith('decoder_'):
                name = file[8:]
                model.add_decoder(name, tf.keras.models.load_model(os.path.join(loc, file)))
        
        return model


#@tf.function
def train_step(
    model: "USIM",
    decoder_name: str,
    x: tf.Tensor,
    y: Tuple[tf.Tensor, tf.Tensor],
    optimizer: Union[str, tf.keras.optimizers.Optimizer],
    loss_fn: Union[str, tf.keras.losses.Loss],
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
    
    if isinstance(loss_fn, str):
        loss_fn = tf.keras.losses.get(loss_fn)

    # Begin tracking gradient
    with tf.GradientTape() as tape:
        logits = model(x, decoder_name=decoder_name, training=True)
        #ys = tf.concat(y,1)
        #ls = tf.concat(logits,1)
        
        loss = loss_fn(y[0], logits[0], sample_weight=weight)
        loss += loss_fn(y[1], logits[1], sample_weight=weight)

    # Apply the gradient:
    weights = model.getTrainable(decoder_name)
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))

    # Update metric if applicable
    metric_fn.update_state(y[0], logits[0])
    metric_fn.update_state(y[1], logits[1])

    return loss
