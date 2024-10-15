from tensorflow.keras.losses import Loss
from tensorflow import Tensor
from typing import Dict, List, Any, Callable

class CustomLoss(Loss):
    def __init__(self, **params):
        '''Initializes the Custom Loss

        Args:
            params: Key-word parameters for the subclassed-loss
        '''
        super(CustomLoss, self).__init__()
        self.params = params

    def call(
            self,
            true_latent: Tensor,
            pred_latent: Tensor,
            true_reconstruction: Tensor,
            pred_reconstruction: Tensor,
            additional_reconstruction: List[Tensor]=[],
            sample_weight=None
            ) -> Tensor:
        '''Calculates the loss

        Args:
            true_latent [Tensor]: The true latent features
            pred_latent [Tensor]: The predicted latent features
            true_reconstruction [Tensor]: The true reconstruction
            pred_reconstruction [Tensor]: The predicted reconstruction
            additional_reconstruction [List[Tensor]]: Additional reconstructions from other decoders
            sample_weight [Tensor]: The sample weights

        Returns:
            Tensor: The loss
        '''
        raise NotImplementedError()

    def get_config(self):
        '''Gets the configuration of the loss
        '''
        return self.params
    
    @classmethod
    def from_config(cls, config):
        '''Creates a CustomLoss from a configuration
        '''
        return cls(config)

class ReconstructionLoss(CustomLoss):
    '''Reconstruction Loss
    Uses one loss function to calculate the loss between the true and predicted reconstruction
    Requires one parameter:
        l1 [Loss]: The loss function to use
    '''
    def __init__(self, **params):
        super().__init__(**params)

        if 'l1' not in self.params:
            raise ValueError('Reconstruction Loss requires an `l1` parameter')

    def call(
            self,
            true_latent: Tensor,
            pred_latent: Tensor,
            true_reconstruction: Tensor,
            pred_reconstruction: Tensor,
            additional_reconstruction: List[Tensor]=[],
            sample_weight=None
            ) -> Tensor:
        '''Calculates the loss

        Args:
            true_latent [Tensor]: The true latent features
            pred_latent [Tensor]: The predicted latent features
            true_reconstruction [Tensor]: The true reconstruction
            pred_reconstruction [Tensor]: The predicted reconstruction
            additional_reconstruction [List[Tensor]]: Additional reconstructions from other decoders
            sample_weight [Tensor]: The sample weights

        Returns:
            Tensor: The loss
        '''
        return self.params['l1'](true_reconstruction, pred_reconstruction, sample_weight)
    
class ReconRegLoss(CustomLoss):
    '''Reconstruction Regularization Loss
    Uses two loss functions to calculate the loss between the true and predicted reconstruction
    Requires three parameters:
        l1 [Loss]: The loss function to use for the reconstruction
        l2 [Loss]: The loss function to use for the regularization
        alpha [float]: The weight of the regularization loss
    '''
    def __init__(self, **params):
        super().__init__(**params)

        if 'l1' not in self.params:
            raise ValueError('Reconstruction Regularization Loss requires an `l1` parameter')
        if 'l2' not in self.params:
            raise ValueError('Reconstruction Regularization Loss requires an `l2` parameter')
        if 'alpha' not in self.params:
            # Default value of 1
            self.params['alpha'] = 1

    def call(
            self,
            true_latent: Tensor,
            pred_latent: Tensor,
            true_reconstruction: Tensor,
            pred_reconstruction: Tensor,
            additional_reconstruction: List[Tensor]=[],
            sample_weight=None
            ) -> Tensor:
        '''Calculates the loss

        Args:
            true_latent [Tensor]: The true latent features
            pred_latent [Tensor]: The predicted latent features
            true_reconstruction [Tensor]: The true reconstruction
            pred_reconstruction [Tensor]: The predicted reconstruction
            additional_reconstruction [List[Tensor]]: Additional reconstructions from other decoders
            sample_weight [Tensor]: The sample weights

        Returns:
            Tensor: The loss
        '''
        loss_reconstruction = self.params['l1'](true_reconstruction, pred_reconstruction, sample_weight)
        loss_regularization = self.params['l2'](true_latent, pred_latent, sample_weight)
        return loss_reconstruction + self.params['alpha'] * loss_regularization
    
class AddingLoss(CustomLoss):
    '''Loss for Adding Additional Decoders
    Uses three loss functions to calculate the loss between the true and predicted reconstruction
    Requires five parameters:
        l1 [Loss]: The loss function to use for the reconstruction
        l2 [Loss]: The loss function to use for the regularization
        l3 [Loss]: The loss function to use for the additional reconstruction
        alpha [float]: The weight of the regularization loss
        beta [float]: The weight of the additional reconstruction loss
    '''
    def __init__(self, **params):
        super().__init__(**params)

        if 'l1' not in self.params:
            raise ValueError('Adding Loss requires an `l1` parameter')
        if 'l2' not in self.params:
            raise ValueError('Adding Loss requires an `l2` parameter')
        if 'l3' not in self.params:
            raise ValueError('Adding Loss requires an `l3` parameter')
        if 'alpha' not in self.params:
            # Default value of 1
            self.params['alpha'] = 1
        if 'beta' not in self.params:
            # Default value of 1
            self.params['beta'] = 1

    def call(
            self,
            true_latent: Tensor,
            pred_latent: Tensor,
            true_reconstruction: Tensor,
            pred_reconstruction: Tensor,
            additional_reconstruction: List[Tensor]=[],
            sample_weight=None
            ) -> Tensor:
        '''Calculates the loss

        Args:
            true_latent [Tensor]: The true latent features
            pred_latent [Tensor]: The predicted latent features
            true_reconstruction [Tensor]: The true reconstruction
            pred_reconstruction [Tensor]: The predicted reconstruction
            additional_reconstruction [List[Tensor]]: Additional reconstructions from other decoders
            sample_weight [Tensor]: The sample weights

        Returns:
            Tensor: The loss
        '''
        loss_reconstruction = self.params['l1'](true_reconstruction, pred_reconstruction, sample_weight)
        loss_regularization = self.params['l2'](true_latent, pred_latent, sample_weight)
        loss_additional = [self.params['l3'](true_reconstruction, rec, sample_weight) for rec in additional_reconstruction]
        loss_additional = sum(loss_additional)/len(loss_additional)
        return loss_reconstruction + self.params['alpha'] * loss_regularization + self.params['beta'] * loss_additional
    
class AddingLoss_SubWrapper(Loss):
    '''Wraps a traditional loss function for standardized use in the AddingLoss
    '''
    def __init__(self, loss: Loss):
        '''Initializes the SubWrapper

        Args:
            loss [Loss]: The loss function to wrap
        '''
        super(AddingLoss_SubWrapper, self).__init__()
        self.loss = loss

    def call(
            self,
            true: Tensor,
            pred: Tensor,
            pred_normalize_func: Callable[[Tensor], Tensor],
            sample_weight=None,
            ) -> Tensor:
        '''Calculates the loss

        Args:
            true [Tensor]: The true values
            pred [Tensor]: The predicted values
            pred_normalize_func [Callable]: The function to normalize the predicted values
            sample_weight [Tensor]: The sample weights

        Returns:
            Tensor: The loss
        '''
        pred = pred_normalize_func(pred)
        return self.loss(true, pred, sample_weight)